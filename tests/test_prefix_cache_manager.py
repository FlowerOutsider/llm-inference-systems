import pytest
import torch

from serving.control_plane.prefix_cache_coordinator import (
    PrefixCacheCoordinator,
)
from serving.control_plane.prefix_cache_manager import PrefixCacheManager
from serving.control_plane.prefix_index import PrefixCacheIndex, PrefixScope
from serving.data_plane.kv_cache import KVCacheError
from serving.data_plane.paged_kv_cache import PagedKVCache


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Prefix cache manager tests require CUDA.",
)


def make_scope() -> PrefixScope:
    return PrefixScope(
        model_id="demo-llm",
        model_revision="v1",
        tokenizer_revision="tokenizer-v1",
        rope_config="rope-theta-10000",
    )


def make_cache(*, num_gpu_blocks: int = 12) -> PagedKVCache:
    return PagedKVCache(
        num_layers=1,
        max_slots=8,
        max_sequence_length=16,
        num_gpu_blocks=num_gpu_blocks,
        block_size=4,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float16,
        device="cuda",
    )


def make_kv(token_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.arange(
        token_count * 2 * 4,
        device="cuda",
        dtype=torch.float16,
    ).reshape(1, token_count, 2, 4)

    return keys, keys + 1000


def write_prefix(
    cache: PagedKVCache,
    *,
    slot_id: int,
    token_ids: list[int],
) -> None:
    keys, values = make_kv(token_count=len(token_ids))

    cache.append(
        layer_idx=0,
        slot_ids=[slot_id],
        keys=keys,
        values=values,
    )


def make_manager(
    *,
    cache: PagedKVCache,
    min_prefix_tokens: int = 4,
    max_entries: int = 2,
    min_free_blocks: int = 0,
) -> tuple[PrefixCacheIndex, PrefixCacheCoordinator, PrefixCacheManager]:
    index = PrefixCacheIndex()
    coordinator = PrefixCacheCoordinator(cache=cache, index=index)
    manager = PrefixCacheManager(
        cache=cache,
        coordinator=coordinator,
        min_prefix_tokens=min_prefix_tokens,
        max_entries=max_entries,
        min_free_blocks=min_free_blocks,
    )
    return index, coordinator, manager


def test_manager_rejects_short_prefix_without_publishing() -> None:
    cache = make_cache()
    index, coordinator, manager = make_manager(
        cache=cache,
        min_prefix_tokens=4,
    )
    source_slot = coordinator.allocate_slots()[0]
    token_ids = [1, 2, 3]

    write_prefix(
        cache,
        slot_id=source_slot,
        token_ids=token_ids,
    )

    result = manager.admit(
        scope=make_scope(),
        token_ids=token_ids,
        source_slot=source_slot,
    )

    assert result.admitted is False
    assert result.reason == "prefix_too_short"
    assert index.lookup(scope=make_scope(), token_ids=token_ids) is None

    stats = manager.stats()
    assert stats["admissions"] == 0
    assert stats["rejected_short_prefixes"] == 1

    coordinator.release_slots([source_slot])


def test_manager_admits_prefix_and_records_cache_hit() -> None:
    cache = make_cache()
    _, coordinator, manager = make_manager(cache=cache)

    source_slot, target_slot = coordinator.allocate_slots(2)
    token_ids = [10, 11, 12, 13, 14, 15]

    write_prefix(
        cache,
        slot_id=source_slot,
        token_ids=token_ids,
    )

    admission = manager.admit(
        scope=make_scope(),
        token_ids=token_ids,
        source_slot=source_slot,
    )

    assert admission.admitted is True
    assert admission.reason == "admitted"
    assert manager.cached_source_slots() == (source_slot,)

    match = manager.attach_longest_prefix(
        scope=make_scope(),
        token_ids=[*token_ids, 99],
        target_slot=target_slot,
    )

    assert match is not None
    assert match.source_slot == source_slot
    assert match.prefix_length == len(token_ids)
    assert cache.length(target_slot) == len(token_ids)

    stats = manager.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 0

    coordinator.release_slots([target_slot])
    manager.evict_lru()

    assert cache.free_slot_count == cache.max_slots


def test_lru_eviction_prefers_least_recently_used_entry() -> None:
    cache = make_cache()
    index, coordinator, manager = make_manager(
        cache=cache,
        max_entries=2,
    )

    source_a, source_b, source_c, target_slot = coordinator.allocate_slots(4)

    tokens_a = [1, 2, 3, 4]
    tokens_b = [5, 6, 7, 8]
    tokens_c = [9, 10, 11, 12]

    write_prefix(cache, slot_id=source_a, token_ids=tokens_a)
    write_prefix(cache, slot_id=source_b, token_ids=tokens_b)
    write_prefix(cache, slot_id=source_c, token_ids=tokens_c)

    assert manager.admit(
        scope=make_scope(),
        token_ids=tokens_a,
        source_slot=source_a,
    ).admitted
    assert manager.admit(
        scope=make_scope(),
        token_ids=tokens_b,
        source_slot=source_b,
    ).admitted

    # 命中 A，使 A 变为最近使用；此时 B 是 LRU。
    match = manager.attach_longest_prefix(
        scope=make_scope(),
        token_ids=[*tokens_a, 99],
        target_slot=target_slot,
    )
    assert match is not None
    assert match.source_slot == source_a

    assert manager.admit(
        scope=make_scope(),
        token_ids=tokens_c,
        source_slot=source_c,
    ).admitted

    assert manager.cached_source_slots() == (source_a, source_c)
    assert index.lookup(scope=make_scope(), token_ids=tokens_b) is None

    with pytest.raises(KVCacheError, match="not allocated"):
        cache.length(source_b)

    assert cache.length(source_a) == len(tokens_a)
    assert cache.length(source_c) == len(tokens_c)

    stats = manager.stats()
    assert stats["evictions"] == 1
    assert stats["entries"] == 2


def test_low_free_block_watermark_evicts_before_admission() -> None:
    cache = make_cache(num_gpu_blocks=4)
    _, coordinator, manager = make_manager(
        cache=cache,
        max_entries=4,
        min_free_blocks=3,
    )

    source_a, source_b = coordinator.allocate_slots(2)
    tokens_a = [1, 2, 3, 4]
    tokens_b = [5, 6, 7, 8]

    write_prefix(cache, slot_id=source_a, token_ids=tokens_a)
    assert cache.free_block_count == 3

    assert manager.admit(
        scope=make_scope(),
        token_ids=tokens_a,
        source_slot=source_a,
    ).admitted

    write_prefix(cache, slot_id=source_b, token_ids=tokens_b)
    assert cache.free_block_count == 2

    admission = manager.admit(
        scope=make_scope(),
        token_ids=tokens_b,
        source_slot=source_b,
    )

    assert admission.admitted is True
    assert manager.cached_source_slots() == (source_b,)
    assert cache.free_block_count >= 3

    stats = manager.stats()
    assert stats["evictions"] == 1
    assert stats["rejected_capacity"] == 0

    with pytest.raises(KVCacheError, match="not allocated"):
        cache.length(source_a)