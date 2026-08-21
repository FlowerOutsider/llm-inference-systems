import pytest
import torch

from serving.control_plane.continuous_batch_scheduler import (
    ContinuousBatchScheduler,
    SchedulerConfig,
)
from serving.control_plane.inference_runtime import (
    InferenceRuntime,
    InferenceRuntimeError,
)
from serving.control_plane.request_lifecycle import RequestLifecycleManager
from serving.control_plane.request_state import InferenceRequest, RequestPhase
from serving.data_plane.paged_kv_cache import KVCacheError, PagedKVCache


def make_cache() -> PagedKVCache:
    return PagedKVCache(
        num_layers=1,
        max_slots=4,
        max_sequence_length=64,
        num_gpu_blocks=16,
        block_size=4,
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


def make_runtime() -> tuple[
    PagedKVCache,
    ContinuousBatchScheduler,
    RequestLifecycleManager,
    InferenceRuntime,
]:
    cache = make_cache()
    scheduler = make_scheduler()
    lifecycle = RequestLifecycleManager(
        cache=cache,
        scheduler=scheduler,
    )
    runtime = InferenceRuntime(
        lifecycle=lifecycle,
        scheduler=scheduler,
    )
    return cache, scheduler, lifecycle, runtime


def test_runtime_advances_request_only_after_work_is_acknowledged() -> None:
    _, scheduler, lifecycle, runtime = make_runtime()
    request = InferenceRequest(
        request_id="runtime-success",
        prompt_token_ids=(1, 2, 3),
        max_new_tokens=2,
    )

    runtime.submit(request)

    prefill_plan = runtime.schedule_once()
    assert len(prefill_plan.prefill) == 1
    assert prefill_plan.prefill[0].request_id == "runtime-success"
    assert prefill_plan.prefill[0].token_start == 0
    assert prefill_plan.prefill[0].token_end == 3
    assert request.phase is RequestPhase.WAITING
    assert runtime.pending_work_item_count == 1

    runtime.acknowledge_prefill(
        "runtime-success",
        token_count=3,
    )
    assert request.phase is RequestPhase.DECODE
    assert runtime.pending_work_item_count == 0

    first_decode_plan = runtime.schedule_once()
    assert len(first_decode_plan.decode) == 1
    assert first_decode_plan.decode[0].request_id == "runtime-success"

    runtime.acknowledge_decode(
        "runtime-success",
        token_id=101,
    )
    assert request.phase is RequestPhase.DECODE
    assert request.generated_token_ids == [101]

    second_decode_plan = runtime.schedule_once()
    assert len(second_decode_plan.decode) == 1

    runtime.acknowledge_decode(
        "runtime-success",
        token_id=102,
    )
    assert request.phase is RequestPhase.FINISHED
    assert lifecycle.active_request_ids() == ()

    with pytest.raises(KeyError):
        scheduler.get_request("runtime-success")


def test_runtime_rejects_schedule_before_previous_work_is_acknowledged() -> None:
    _, _, _, runtime = make_runtime()
    request = InferenceRequest(
        request_id="inflight-work",
        prompt_token_ids=(1, 2, 3),
        max_new_tokens=1,
    )

    runtime.submit(request)
    runtime.schedule_once()

    with pytest.raises(InferenceRuntimeError, match="unacknowledged"):
        runtime.schedule_once()


def test_runtime_cancel_clears_inflight_work_and_releases_slot() -> None:
    cache, scheduler, lifecycle, runtime = make_runtime()
    request = InferenceRequest(
        request_id="cancel-runtime",
        prompt_token_ids=(1, 2, 3),
        max_new_tokens=2,
    )

    admission = runtime.submit(request)
    runtime.schedule_once()

    runtime.cancel("cancel-runtime")

    assert request.phase is RequestPhase.CANCELLED
    assert runtime.pending_work_item_count == 0
    assert lifecycle.active_request_ids() == ()

    with pytest.raises(KeyError):
        scheduler.get_request("cancel-runtime")

    with pytest.raises(KVCacheError, match="not allocated"):
        cache.length(admission.slot_id)


def test_runtime_fail_clears_inflight_work_and_releases_slot() -> None:
    cache, scheduler, lifecycle, runtime = make_runtime()
    request = InferenceRequest(
        request_id="fail-runtime",
        prompt_token_ids=(1, 2, 3),
        max_new_tokens=2,
    )

    admission = runtime.submit(request)
    runtime.schedule_once()

    runtime.fail(
        "fail-runtime",
        reason="vLLM worker timeout",
    )

    assert request.phase is RequestPhase.FAILED
    assert request.failure_reason == "vLLM worker timeout"
    assert runtime.pending_work_item_count == 0
    assert lifecycle.active_request_ids() == ()

    with pytest.raises(KeyError):
        scheduler.get_request("fail-runtime")

    with pytest.raises(KVCacheError, match="not allocated"):
        cache.length(admission.slot_id)


def test_runtime_rejects_acknowledgement_for_unknown_work() -> None:
    _, _, _, runtime = make_runtime()

    with pytest.raises(InferenceRuntimeError, match="no pending prefill"):
        runtime.acknowledge_prefill(
            "unknown-request",
            token_count=1,
        )

    with pytest.raises(InferenceRuntimeError, match="no pending decode"):
        runtime.acknowledge_decode(
            "unknown-request",
            token_id=42,
        )