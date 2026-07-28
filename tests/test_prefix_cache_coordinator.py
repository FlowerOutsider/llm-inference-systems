import pytest
import torch

from serving.control_plane.prefix_cache_coordinator import (
    PrefixCacheCoordinator,
)
from serving.control_plane.prefix_index import PrefixCacheIndex, PrefixScope
from serving.data_plane.paged_kv_cache import PagedKVCache


def make_scope(model_revision: str = "v1") -> PrefixScope:
    return PrefixScope(
        model_id="demo-llm",
        model_revision=model_revision,
        tokenizer_revision="tokenizer-v1",
        rope_config="rope-theta-10000",
    )


def make_cache() -> PagedKVCache:
    return PagedKVCache(
        num_layers=1,
        max_slots=4,
        max_sequence_length=16,
        num_gpu_blocks=8,
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
        dtype=torch.float32,
    ).reshape(1, token_count, 2, 4).to(torch.float16)

    return keys, keys + 100


def test_coordinator_attaches_longest_prefix_and_releases_source_safely() -> None:
    cache = make_cache()
    index = PrefixCacheIndex()
    coordinator = PrefixCacheCoordinator(cache=cache, index=index)

    source_slot, target_slot = coordinator.allocate_slots(2)
    token_ids = [10, 11, 12, 13, 14, 15]
    keys, values = make_kv(token_count=len(token_ids))

    cache.append(
        layer_idx=0,
        slot_ids=[source_slot],
        keys=keys,
        values=values,
    )

    scope = make_scope()
    coordinator.publish_prefix(
        scope=scope,
        token_ids=token_ids,
        source_slot=source_slot,
    )

    match = coordinator.attach_longest_prefix(
        scope=scope,
        token_ids=[*token_ids, 99],
        target_slot=target_slot,
    )

    assert match is not None
    assert match.source_slot == source_slot
    assert match.prefix_length == len(token_ids)
    assert cache.length(target_slot) == len(token_ids)
    assert (
        cache.block_table(target_slot).cpu().tolist()
        == cache.block_table(source_slot).cpu().tolist()
    )

    shared_blocks = cache.block_table(source_slot).cpu().tolist()
    assert all(cache.block_refcount(block_id) == 2 for block_id in shared_blocks)

    coordinator.release_slots([source_slot])

    assert index.lookup(scope=scope, token_ids=[*token_ids, 99]) is None
    assert cache.length(target_slot) == len(token_ids)
    assert all(cache.block_refcount(block_id) == 1 for block_id in shared_blocks)


def test_coordinator_rejects_uncommitted_or_mismatched_prefix_metadata() -> None:
    cache = make_cache()
    coordinator = PrefixCacheCoordinator(
        cache=cache,
        index=PrefixCacheIndex(),
    )

    source_slot = coordinator.allocate_slots()[0]
    scope = make_scope()

    with pytest.raises(ValueError, match="does not match cache length"):
        coordinator.publish_prefix(
            scope=scope,
            token_ids=[1],
            source_slot=source_slot,
        )

    target_slot = coordinator.allocate_slots()[0]
    match = coordinator.attach_longest_prefix(
        scope=scope,
        token_ids=[1, 2, 3],
        target_slot=target_slot,
    )

    assert match is None
    assert cache.length(target_slot) == 0