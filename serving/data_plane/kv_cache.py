from __future__ import annotations

from collections.abc import Sequence

import torch  


class KVCacheError(RuntimeError):
    """Base exception for KV cache lifecycle and validation failures."""


class KVCacheCapacityError(KVCacheError):
    """Raised when a slot allocation or append would exceed cache capacity."""


class ContiguousKVCache:
    """
    GPU-resident, preallocated KV Cache with slot-based request lifecycle.

    Layout:
        [num_layers, max_slots, max_sequence_length, num_kv_heads, head_dim]

    This is intentionally a contiguous-memory baseline. It avoids allocating
    and concatenating historical KV tensors during decode, but it still suffers
    from the internal fragmentation that PagedAttention later addresses.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        max_slots: int,
        max_sequence_length: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        self._validate_positive("num_layers", num_layers)
        self._validate_positive("max_slots", max_slots)
        self._validate_positive("max_sequence_length", max_sequence_length)
        self._validate_positive("num_kv_heads", num_kv_heads)
        self._validate_positive("head_dim", head_dim)

        self.num_layers = num_layers
        self.max_slots = max_slots
        self.max_sequence_length = max_sequence_length
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        
        requested_device = torch.device(device)

        if requested_device.type == "cuda" and requested_device.index is None:
            requested_device = torch.device("cuda", torch.cuda.current_device())

        self.device = requested_device

        shape = (
            num_layers,
            max_slots,
            max_sequence_length,
            num_kv_heads,
            head_dim,
        )

        self._keys = torch.empty(shape, dtype=dtype, device=self.device)
        self._values = torch.empty_like(self._keys)

        # 生命周期元数据保留在 CPU：调度器通常在 CPU 侧管理请求和 slot。
        self._lengths = [0] * max_slots
        self._active_slots: set[int] = set()

    @staticmethod
    def _validate_positive(name: str, value: int) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    @property
    def active_slot_count(self) -> int:
        return len(self._active_slots)

    @property
    def free_slot_count(self) -> int:
        return self.max_slots - self.active_slot_count

    @property
    def physical_memory_bytes(self) -> int:
        return self._keys.numel() * self._keys.element_size() * 2

    @property
    def physical_memory_mib(self) -> float:
        return self.physical_memory_bytes / (1024 * 1024)

    @property
    def bytes_per_token_all_layers(self) -> int:
        return (
            self.num_layers
            * 2
            * self.num_kv_heads
            * self.head_dim
            * self._keys.element_size()
        )

    def logical_memory_bytes(self) -> int:
        return sum(self._lengths) * self.bytes_per_token_all_layers

    def length(self, slot_id: int) -> int:
        self._validate_slot_ids([slot_id])
        return self._lengths[slot_id]

    def allocate(self, count: int = 1) -> list[int]:
        self._validate_positive("count", count)

        if count > self.free_slot_count:
            raise KVCacheCapacityError(
                f"requested {count} slots, but only {self.free_slot_count} are free"
            )

        allocated: list[int] = []
        for slot_id in range(self.max_slots):
            if slot_id not in self._active_slots:
                self._active_slots.add(slot_id)
                self._lengths[slot_id] = 0
                allocated.append(slot_id)

                if len(allocated) == count:
                    return allocated

        raise AssertionError("free-slot accounting is inconsistent")

    def release(self, slot_ids: Sequence[int]) -> None:
        normalized_slot_ids = self._validate_slot_ids(slot_ids)

        for slot_id in normalized_slot_ids:
            self._active_slots.remove(slot_id)
            self._lengths[slot_id] = 0

    def append(
        self,
        *,
        layer_idx: int,
        slot_ids: Sequence[int],
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """
        Append an equal number of tokens to each requested slot.

        keys and values must have shape:
            [batch_size, token_count, num_kv_heads, head_dim]
        """
        self._validate_layer_idx(layer_idx)
        normalized_slot_ids = self._validate_slot_ids(slot_ids)
        self._validate_append_tensors(keys, values, len(normalized_slot_ids))

        token_count = keys.size(1)
        start_positions = [self._lengths[slot_id] for slot_id in normalized_slot_ids]

        for slot_id, start_position in zip(normalized_slot_ids, start_positions):
            end_position = start_position + token_count
            if end_position > self.max_sequence_length:
                raise KVCacheCapacityError(
                    f"slot {slot_id} append exceeds capacity: "
                    f"{start_position} + {token_count} > {self.max_sequence_length}"
                )

        slot_index = torch.tensor(
            normalized_slot_ids,
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(1)

        position_index = torch.tensor(
            start_positions,
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(1) + torch.arange(
            token_count,
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0)

        self._keys[layer_idx, slot_index, position_index] = keys
        self._values[layer_idx, slot_index, position_index] = values

        for slot_id in normalized_slot_ids:
            self._lengths[slot_id] += token_count

    def get_kv(
        self,
        *,
        layer_idx: int,
        slot_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return views of valid K and V positions for a single active request.

        Returned shape:
            [sequence_length, num_kv_heads, head_dim]
        """
        self._validate_layer_idx(layer_idx)
        self._validate_slot_ids([slot_id])

        sequence_length = self._lengths[slot_id]
        return (
            self._keys[layer_idx, slot_id, :sequence_length],
            self._values[layer_idx, slot_id, :sequence_length],
        )

    def summary(self) -> dict[str, int | float]:
        return {
            "active_slots": self.active_slot_count,
            "free_slots": self.free_slot_count,
            "logical_tokens": sum(self._lengths),
            "physical_memory_mib": self.physical_memory_mib,
            "logical_memory_mib": self.logical_memory_bytes() / (1024 * 1024),
        }

    def _validate_layer_idx(self, layer_idx: int) -> None:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(
                f"layer_idx must be in [0, {self.num_layers}), got {layer_idx}"
            )

    def _validate_slot_ids(self, slot_ids: Sequence[int]) -> list[int]:
        if not slot_ids:
            raise ValueError("slot_ids must not be empty")

        normalized_slot_ids = [int(slot_id) for slot_id in slot_ids]

        if len(set(normalized_slot_ids)) != len(normalized_slot_ids):
            raise ValueError("slot_ids must not contain duplicates")

        for slot_id in normalized_slot_ids:
            if not 0 <= slot_id < self.max_slots:
                raise IndexError(
                    f"slot_id must be in [0, {self.max_slots}), got {slot_id}"
                )

            if slot_id not in self._active_slots:
                raise KVCacheError(f"slot {slot_id} is not allocated")

        return normalized_slot_ids

    def _validate_append_tensors(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        expected_batch_size: int,
    ) -> None:
        if keys.shape != values.shape:
            raise ValueError(
                f"keys and values must have identical shapes, got "
                f"{tuple(keys.shape)} and {tuple(values.shape)}"
            )

        if keys.ndim != 4:
            raise ValueError(
                "keys and values must have shape "
                "[batch_size, token_count, num_kv_heads, head_dim]"
            )

        batch_size, token_count, num_kv_heads, head_dim = keys.shape

        if batch_size != expected_batch_size:
            raise ValueError(
                f"batch size {batch_size} does not match {expected_batch_size} slot IDs"
            )

        if token_count <= 0:
            raise ValueError("token_count must be positive")

        if num_kv_heads != self.num_kv_heads or head_dim != self.head_dim:
            raise ValueError(
                "KV tensor head shape does not match cache configuration: "
                f"expected ({self.num_kv_heads}, {self.head_dim}), "
                f"got ({num_kv_heads}, {head_dim})"
            )

        if keys.dtype != self.dtype:
            raise ValueError(f"expected dtype {self.dtype}, got {keys.dtype}")

        if keys.device != self.device:
            raise ValueError(f"expected device {self.device}, got {keys.device}")