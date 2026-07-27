import pytest
import torch

from serving.data_plane.kv_cache import (
    ContiguousKVCache,
    KVCacheCapacityError,
    KVCacheError,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="KV Cache tests require CUDA.",
)


def make_cache(
    *,
    max_slots: int = 4,
    max_sequence_length: int = 8,
) -> ContiguousKVCache:
    return ContiguousKVCache(
        num_layers=2,
        max_slots=max_slots,
        max_sequence_length=max_sequence_length,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float16,
        device="cuda",
    )


def make_kv(
    batch_size: int,
    token_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    element_count = batch_size * token_count * 2 * 4

    keys = torch.arange(
        element_count,
        device="cuda",
        dtype=torch.float16,
    ).reshape(batch_size, token_count, 2, 4)

    values = keys + 1000
    return keys, values


def test_append_and_read_returns_written_tensor() -> None:
    cache = make_cache()
    slots = cache.allocate(2)
    keys, values = make_kv(batch_size=2, token_count=3)

    cache.append(
        layer_idx=0,
        slot_ids=slots,
        keys=keys,
        values=values,
    )

    stored_keys, stored_values = cache.get_kv(layer_idx=0, slot_id=slots[0])

    assert cache.length(slots[0]) == 3
    torch.testing.assert_close(stored_keys, keys[0])
    torch.testing.assert_close(stored_values, values[0])


def test_append_preserves_existing_history() -> None:
    cache = make_cache()
    slot = cache.allocate()[0]

    first_keys, first_values = make_kv(batch_size=1, token_count=2)
    second_keys, second_values = make_kv(batch_size=1, token_count=1)

    cache.append(
        layer_idx=0,
        slot_ids=[slot],
        keys=first_keys,
        values=first_values,
    )
    cache.append(
        layer_idx=0,
        slot_ids=[slot],
        keys=second_keys,
        values=second_values,
    )

    stored_keys, stored_values = cache.get_kv(layer_idx=0, slot_id=slot)

    torch.testing.assert_close(stored_keys[:2], first_keys[0])
    torch.testing.assert_close(stored_keys[2:], second_keys[0])
    torch.testing.assert_close(stored_values[:2], first_values[0])
    torch.testing.assert_close(stored_values[2:], second_values[0])
    assert cache.length(slot) == 3


def test_append_over_capacity_does_not_change_length() -> None:
    cache = make_cache(max_sequence_length=3)
    slot = cache.allocate()[0]

    first_keys, first_values = make_kv(batch_size=1, token_count=2)
    cache.append(
        layer_idx=0,
        slot_ids=[slot],
        keys=first_keys,
        values=first_values,
    )

    overflowing_keys, overflowing_values = make_kv(
        batch_size=1,
        token_count=2,
    )

    with pytest.raises(KVCacheCapacityError, match="exceeds capacity"):
        cache.append(
            layer_idx=0,
            slot_ids=[slot],
            keys=overflowing_keys,
            values=overflowing_values,
        )

    assert cache.length(slot) == 2


def test_released_slot_cannot_be_read_or_written_and_can_be_reused() -> None:
    cache = make_cache()
    slot = cache.allocate()[0]
    cache.release([slot])

    keys, values = make_kv(batch_size=1, token_count=1)

    with pytest.raises(KVCacheError, match="not allocated"):
        cache.get_kv(layer_idx=0, slot_id=slot)

    with pytest.raises(KVCacheError, match="not allocated"):
        cache.append(
            layer_idx=0,
            slot_ids=[slot],
            keys=keys,
            values=values,
        )

    reused_slot = cache.allocate()[0]

    assert reused_slot == slot
    assert cache.length(reused_slot) == 0


def test_cuda_default_device_is_normalized_to_explicit_index() -> None:
    cache = make_cache()

    assert cache.device == torch.device("cuda", torch.cuda.current_device())