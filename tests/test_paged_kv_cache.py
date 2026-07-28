import pytest
import torch

from serving.data_plane.kv_cache import KVCacheCapacityError, KVCacheError
from serving.data_plane.paged_kv_cache import PagedKVCache


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Paged KV Cache tests require CUDA.",
)


def make_cache(
    *,
    num_layers: int = 1,
    num_gpu_blocks: int = 8,
    block_size: int = 4,
) -> PagedKVCache:
    return PagedKVCache(
        num_layers=num_layers,
        max_slots=3,
        max_sequence_length=16,
        num_gpu_blocks=num_gpu_blocks,
        block_size=block_size,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float16,
        device="cuda",
    )

def make_kv(
    batch_size: int,
    token_count: int,
    value_offset: float = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    element_count = batch_size * token_count * 2 * 4

    keys = torch.arange(
        element_count,
        device="cuda",
        dtype=torch.float16,
    ).reshape(batch_size, token_count, 2, 4) + value_offset

    values = keys + 1000
    return keys, values


def test_append_across_blocks_and_gather_in_logical_order() -> None:
    cache = make_cache(block_size=4)
    slot = cache.allocate()[0]
    keys, values = make_kv(batch_size=1, token_count=6)

    cache.append(
        layer_idx=0,
        slot_ids=[slot],
        keys=keys,
        values=values,
    )

    stored_keys, stored_values = cache.get_kv(layer_idx=0, slot_id=slot)
    block_table = cache.block_table(slot).cpu().tolist()

    assert cache.length(slot) == 6
    assert len(block_table) == 2
    assert len(set(block_table)) == 2
    assert all(0 <= block_id < cache.num_gpu_blocks for block_id in block_table)
    torch.testing.assert_close(stored_keys, keys[0])
    torch.testing.assert_close(stored_values, values[0])


def test_append_reuses_partially_filled_last_block() -> None:
    cache = make_cache(block_size=4)
    slot = cache.allocate()[0]

    first_keys, first_values = make_kv(batch_size=1, token_count=3)
    second_keys, second_values = make_kv(
        batch_size=1,
        token_count=2,
        value_offset=100,
    )

    cache.append(
        layer_idx=0,
        slot_ids=[slot],
        keys=first_keys,
        values=first_values,
    )
    first_block_table = cache.block_table(slot).cpu().tolist()

    cache.append(
        layer_idx=0,
        slot_ids=[slot],
        keys=second_keys,
        values=second_values,
    )
    second_block_table = cache.block_table(slot).cpu().tolist()

    stored_keys, stored_values = cache.get_kv(layer_idx=0, slot_id=slot)

    assert len(first_block_table) == 1
    assert len(second_block_table) == 2
    assert second_block_table[0] == first_block_table[0]
    assert cache.length(slot) == 5
    torch.testing.assert_close(stored_keys[:3], first_keys[0])
    torch.testing.assert_close(stored_keys[3:], second_keys[0])
    torch.testing.assert_close(stored_values[:3], first_values[0])
    torch.testing.assert_close(stored_values[3:], second_values[0])


def test_block_exhaustion_is_checked_before_mutating_cache() -> None:
    cache = make_cache(num_gpu_blocks=2, block_size=4)
    first_slot, second_slot = cache.allocate(2)

    first_keys, first_values = make_kv(batch_size=1, token_count=5)
    cache.append(
        layer_idx=0,
        slot_ids=[first_slot],
        keys=first_keys,
        values=first_values,
    )

    overflowing_keys, overflowing_values = make_kv(
        batch_size=1,
        token_count=1,
    )

    with pytest.raises(KVCacheCapacityError, match="free blocks"):
        cache.append(
            layer_idx=0,
            slot_ids=[second_slot],
            keys=overflowing_keys,
            values=overflowing_values,
        )

    assert cache.length(first_slot) == 5
    assert cache.length(second_slot) == 0
    assert cache.allocated_block_count == 2
    assert cache.free_block_count == 0


def test_release_returns_blocks_to_pool_for_future_requests() -> None:
    cache = make_cache(num_gpu_blocks=2, block_size=4)
    first_slot = cache.allocate()[0]
    keys, values = make_kv(batch_size=1, token_count=5)

    cache.append(
        layer_idx=0,
        slot_ids=[first_slot],
        keys=keys,
        values=values,
    )

    assert cache.free_block_count == 0

    cache.release([first_slot])

    assert cache.free_block_count == 2
    assert cache.allocated_block_count == 0

    second_slot = cache.allocate()[0]
    next_keys, next_values = make_kv(batch_size=1, token_count=4)

    cache.append(
        layer_idx=0,
        slot_ids=[second_slot],
        keys=next_keys,
        values=next_values,
    )

    assert cache.length(second_slot) == 4
    assert cache.allocated_block_count == 1

def test_multilayer_append_commits_length_only_after_all_layers_write() -> None:
    cache = make_cache(num_layers=2)
    slot = cache.allocate()[0]

    layer_zero_keys, layer_zero_values = make_kv(
        batch_size=1,
        token_count=2,
        value_offset=0,
    )
    layer_one_keys, layer_one_values = make_kv(
        batch_size=1,
        token_count=2,
        value_offset=500,
    )

    reservation = cache.begin_append(slot_ids=[slot], token_count=2)

    cache.write_layer(
        layer_idx=0,
        reservation=reservation,
        keys=layer_zero_keys,
        values=layer_zero_values,
    )
    cache.write_layer(
        layer_idx=1,
        reservation=reservation,
        keys=layer_one_keys,
        values=layer_one_values,
    )

    assert cache.length(slot) == 0

    cache.commit_append(reservation)

    assert cache.length(slot) == 2

    stored_layer_zero_keys, stored_layer_zero_values = cache.get_kv(
        layer_idx=0,
        slot_id=slot,
    )
    stored_layer_one_keys, stored_layer_one_values = cache.get_kv(
        layer_idx=1,
        slot_id=slot,
    )

    torch.testing.assert_close(stored_layer_zero_keys, layer_zero_keys[0])
    torch.testing.assert_close(stored_layer_zero_values, layer_zero_values[0])
    torch.testing.assert_close(stored_layer_one_keys, layer_one_keys[0])
    torch.testing.assert_close(stored_layer_one_values, layer_one_values[0])


def test_incomplete_multilayer_append_cannot_commit_and_can_abort() -> None:
    cache = make_cache(num_layers=2, num_gpu_blocks=2, block_size=4)
    slot = cache.allocate()[0]
    keys, values = make_kv(batch_size=1, token_count=2)

    reservation = cache.begin_append(slot_ids=[slot], token_count=2)

    cache.write_layer(
        layer_idx=0,
        reservation=reservation,
        keys=keys,
        values=values,
    )

    with pytest.raises(KVCacheError, match="all layers"):
        cache.commit_append(reservation)

    cache.abort_append(reservation)

    assert cache.length(slot) == 0
    assert cache.allocated_block_count == 0
    assert cache.free_block_count == 2



def test_fork_full_block_prefix_shares_physical_block() -> None:
    cache = make_cache(block_size=4)
    source_slot, target_slot = cache.allocate(2)

    keys, values = make_kv(batch_size=1, token_count=6)
    cache.append(
        layer_idx=0,
        slot_ids=[source_slot],
        keys=keys,
        values=values,
    )

    source_blocks = cache.block_table(source_slot).cpu().tolist()

    cache.fork_prefix(
        source_slot=source_slot,
        target_slot=target_slot,
        prefix_length=4,
    )

    target_blocks = cache.block_table(target_slot).cpu().tolist()

    assert cache.length(source_slot) == 6
    assert cache.length(target_slot) == 4
    assert source_blocks[0] == target_blocks[0]
    assert cache.block_refcount(source_blocks[0]) == 2

    # 共享 block 不应额外分配物理显存。
    assert cache.allocated_block_count == 2


def test_shared_block_is_reclaimed_only_after_last_owner_releases() -> None:
    cache = make_cache(block_size=4)
    source_slot, target_slot = cache.allocate(2)

    keys, values = make_kv(batch_size=1, token_count=6)
    cache.append(
        layer_idx=0,
        slot_ids=[source_slot],
        keys=keys,
        values=values,
    )

    source_blocks = cache.block_table(source_slot).cpu().tolist()
    shared_block = source_blocks[0]

    cache.fork_prefix(
        source_slot=source_slot,
        target_slot=target_slot,
        prefix_length=4,
    )

    cache.release([target_slot])

    assert cache.block_refcount(shared_block) == 1
    assert cache.allocated_block_count == 2

    cache.release([source_slot])

    assert cache.free_block_count == cache.num_gpu_blocks
    assert cache.allocated_block_count == 0




def test_append_after_partial_prefix_fork_uses_cow_and_preserves_source() -> None:
    cache = make_cache(block_size=4, num_gpu_blocks=4)
    source_slot, target_slot = cache.allocate(2)

    source_keys, source_values = make_kv(
        batch_size=1,
        token_count=6,
        value_offset=0,
    )
    cache.append(
        layer_idx=0,
        slot_ids=[source_slot],
        keys=source_keys,
        values=source_values,
    )

    source_blocks = cache.block_table(source_slot).cpu().tolist()

    cache.fork_prefix(
        source_slot=source_slot,
        target_slot=target_slot,
        prefix_length=6,
    )

    appended_keys, appended_values = make_kv(
        batch_size=1,
        token_count=1,
        value_offset=1000,
    )
    cache.append(
        layer_idx=0,
        slot_ids=[target_slot],
        keys=appended_keys,
        values=appended_values,
    )

    target_blocks = cache.block_table(target_slot).cpu().tolist()

    assert cache.length(source_slot) == 6
    assert cache.length(target_slot) == 7

    assert target_blocks[0] == source_blocks[0]
    assert target_blocks[1] != source_blocks[1]
    assert cache.block_refcount(source_blocks[1]) == 1
    assert cache.block_refcount(target_blocks[1]) == 1

    source_stored_keys, source_stored_values = cache.get_kv(
        layer_idx=0,
        slot_id=source_slot,
    )
    target_stored_keys, target_stored_values = cache.get_kv(
        layer_idx=0,
        slot_id=target_slot,
    )

    torch.testing.assert_close(source_stored_keys, source_keys[0])
    torch.testing.assert_close(source_stored_values, source_values[0])

    expected_target_keys = torch.cat(
        [source_keys[0], appended_keys[0]],
        dim=0,
    )
    expected_target_values = torch.cat(
        [source_values[0], appended_values[0]],
        dim=0,
    )
    torch.testing.assert_close(target_stored_keys, expected_target_keys)
    torch.testing.assert_close(target_stored_values, expected_target_values)


def test_abort_append_restores_shared_tail_block_after_cow() -> None:
    cache = make_cache(block_size=4, num_gpu_blocks=4)
    source_slot, target_slot = cache.allocate(2)

    keys, values = make_kv(batch_size=1, token_count=6)
    cache.append(
        layer_idx=0,
        slot_ids=[source_slot],
        keys=keys,
        values=values,
    )

    source_blocks = cache.block_table(source_slot).cpu().tolist()

    cache.fork_prefix(
        source_slot=source_slot,
        target_slot=target_slot,
        prefix_length=6,
    )

    reservation = cache.begin_append(
        slot_ids=[target_slot],
        token_count=1,
    )

    temporary_target_blocks = cache.block_table(target_slot).cpu().tolist()
    assert temporary_target_blocks[1] != source_blocks[1]
    assert cache.block_refcount(source_blocks[1]) == 1
    assert cache.allocated_block_count == 3

    cache.abort_append(reservation)

    restored_target_blocks = cache.block_table(target_slot).cpu().tolist()
    assert restored_target_blocks == source_blocks
    assert cache.length(target_slot) == 6
    assert cache.block_refcount(source_blocks[1]) == 2
    assert cache.allocated_block_count == 2