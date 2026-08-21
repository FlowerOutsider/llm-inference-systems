import pytest
import torch

from serving.control_plane.continuous_batch_scheduler import (
    ContinuousBatchScheduler,
    SchedulerConfig,
)
from serving.control_plane.prefix_cache_coordinator import PrefixCacheCoordinator
from serving.control_plane.prefix_cache_manager import PrefixCacheManager
from serving.control_plane.prefix_index import PrefixCacheIndex, PrefixScope
from serving.control_plane.request_lifecycle import (
    RequestAdmissionError,
    RequestLifecycleManager,
)
from serving.control_plane.request_state import (
    InferenceRequest,
    RequestPhase,
)
from serving.data_plane.paged_kv_cache import KVCacheError, PagedKVCache


def make_cache(
    *,
    num_slots: int = 4,
    num_gpu_blocks: int = 16,
    block_size: int = 4,
) -> PagedKVCache:
    return PagedKVCache(
        num_layers=1,
        max_slots=num_slots,
        max_sequence_length=64,
        num_gpu_blocks=num_gpu_blocks,
        block_size=block_size,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float16,
        device="cuda",
    )


def make_scheduler() -> ContinuousBatchScheduler:
    return ContinuousBatchScheduler(
        SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=16,
            prefill_chunk_size=8,
        )
    )


def make_scope() -> PrefixScope:
    return PrefixScope(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        model_revision="main",
        tokenizer_revision="main",
        rope_config="default",
    )


def make_kv(token_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.ones(
        (1, token_count, 2, 4),
        dtype=torch.float16,
        device="cuda",
    )
    values = torch.full(
        (1, token_count, 2, 4),
        fill_value=2.0,
        dtype=torch.float16,
        device="cuda",
    )
    return keys, values


def make_prefix_manager(cache: PagedKVCache) -> PrefixCacheManager:
    index = PrefixCacheIndex()
    coordinator = PrefixCacheCoordinator(
        cache=cache,
        index=index,
    )
    return PrefixCacheManager(
        cache=cache,
        coordinator=coordinator,
        min_prefix_tokens=4,
        max_entries=4,
        min_free_blocks=0,
    )


def test_submit_allocates_real_kv_slot_and_enters_scheduler() -> None:
    cache = make_cache()
    scheduler = make_scheduler()
    manager = RequestLifecycleManager(
        cache=cache,
        scheduler=scheduler,
    )
    request = InferenceRequest(
        request_id="request-a",
        prompt_token_ids=(1, 2, 3),
        max_new_tokens=2,
    )

    admission = manager.submit(request)

    assert admission.slot_id == manager.slot_for("request-a")
    assert cache.length(admission.slot_id) == 0
    assert scheduler.active_request_ids() == ("request-a",)


def test_cancel_releases_real_kv_slot() -> None:
    cache = make_cache()
    scheduler = make_scheduler()
    manager = RequestLifecycleManager(
        cache=cache,
        scheduler=scheduler,
    )
    request = InferenceRequest(
        request_id="cancel-me",
        prompt_token_ids=(1, 2, 3),
        max_new_tokens=2,
    )

    slot_id = manager.submit(request).slot_id
    manager.cancel("cancel-me")

    assert request.phase is RequestPhase.CANCELLED
    assert scheduler.active_request_ids() == ()

    with pytest.raises(KVCacheError, match="not allocated"):
        cache.length(slot_id)

    with pytest.raises(KeyError, match="cancel-me"):
        manager.slot_for("cancel-me")


def test_finalize_releases_slot_only_after_request_finished() -> None:
    cache = make_cache()
    scheduler = make_scheduler()
    manager = RequestLifecycleManager(
        cache=cache,
        scheduler=scheduler,
    )
    request = InferenceRequest(
        request_id="finish-me",
        prompt_token_ids=(10, 11),
        max_new_tokens=2,
    )

    slot_id = manager.submit(request).slot_id

    scheduler.mark_prefill_executed("finish-me", token_count=2)
    scheduler.mark_decode_executed("finish-me", token_id=101)
    scheduler.mark_decode_executed("finish-me", token_id=102)

    assert request.phase is RequestPhase.FINISHED

    manager.finalize("finish-me")

    with pytest.raises(KVCacheError, match="not allocated"):
        cache.length(slot_id)


def test_prefix_hit_reuses_kv_and_skips_reused_prefill_tokens() -> None:
    cache = make_cache()
    scheduler = make_scheduler()
    prefix_manager = make_prefix_manager(cache)
    lifecycle = RequestLifecycleManager(
        cache=cache,
        scheduler=scheduler,
        prefix_cache_manager=prefix_manager,
    )
    scope = make_scope()

    source_slot = cache.allocate(1)[0]
    keys, values = make_kv(token_count=4)
    cache.append(
        layer_idx=0,
        slot_ids=[source_slot],
        keys=keys,
        values=values,
    )

    prefix_manager.admit(
        scope=scope,
        token_ids=(1, 2, 3, 4),
        source_slot=source_slot,
    )

    request = InferenceRequest(
        request_id="prefix-hit",
        prompt_token_ids=(1, 2, 3, 4, 5, 6),
        max_new_tokens=2,
    )
    admission = lifecycle.submit(
        request,
        prefix_scope=scope,
    )

    assert admission.reused_prefix_tokens == 4
    assert request.phase is RequestPhase.PREFILL
    assert request.prefill_offset == 4
    assert cache.length(admission.slot_id) == 4

    plan = scheduler.schedule()
    assert [(item.request_id, item.token_start, item.token_end) for item in plan.prefill] == [
        ("prefix-hit", 4, 6)
    ]

    lifecycle.cancel("prefix-hit")

    assert cache.block_refcount(cache.block_table(source_slot)[0].item()) == 1


def test_slot_exhaustion_rejects_request_without_submitting_to_scheduler() -> None:
    cache = make_cache(num_slots=1)
    scheduler = make_scheduler()
    manager = RequestLifecycleManager(
        cache=cache,
        scheduler=scheduler,
    )

    manager.submit(
        InferenceRequest(
            request_id="first",
            prompt_token_ids=(1,),
            max_new_tokens=1,
        )
    )

    with pytest.raises(RequestAdmissionError, match="KV slot"):
        manager.submit(
            InferenceRequest(
                request_id="second",
                prompt_token_ids=(2,),
                max_new_tokens=1,
            )
        )

    assert scheduler.active_request_ids() == ("first",)


def test_fail_releases_slot_and_removes_request_from_scheduler() -> None:
    cache = make_cache()
    scheduler = make_scheduler()
    manager = RequestLifecycleManager(
        cache=cache,
        scheduler=scheduler,
    )
    request = InferenceRequest(
        request_id="worker-failure",
        prompt_token_ids=(1, 2, 3),
        max_new_tokens=2,
    )

    slot_id = manager.submit(request).slot_id
    manager.fail("worker-failure", reason="vLLM worker timeout")

    assert request.phase is RequestPhase.FAILED
    assert request.failure_reason == "vLLM worker timeout"
    assert manager.active_request_ids() == ()

    with pytest.raises(KeyError, match="worker-failure"):
        scheduler.get_request("worker-failure")

    with pytest.raises(KVCacheError, match="not allocated"):
        cache.length(slot_id)