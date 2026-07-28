from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from serving.data_plane.kv_cache import KVCacheCapacityError, KVCacheError


@dataclass
class AppendReservation:
    """A pending multi-layer append transaction."""

    reservation_id: int
    slot_ids: tuple[int, ...]
    start_positions: tuple[int, ...]
    token_count: int
    new_blocks_by_slot: dict[int, tuple[int, ...]]
    written_layers: set[int] = field(default_factory=set)


class PagedKVCache:
    """
    GPU-resident KV pool with fixed-size blocks and transactional append.

    Each token range is first reserved once, then written by every transformer
    layer at identical logical positions. Sequence lengths advance only when the
    reservation has been written by all layers and committed successfully.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        max_slots: int,
        max_sequence_length: int,
        num_gpu_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        for name, value in (
            ("num_layers", num_layers),
            ("max_slots", max_slots),
            ("max_sequence_length", max_sequence_length),
            ("num_gpu_blocks", num_gpu_blocks),
            ("block_size", block_size),
            ("num_kv_heads", num_kv_heads),
            ("head_dim", head_dim),
        ):
            self._validate_positive(name, value)

        self.num_layers = num_layers
        self.max_slots = max_slots
        self.max_sequence_length = max_sequence_length
        self.num_gpu_blocks = num_gpu_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        requested_device = torch.device(device)
        if requested_device.type == "cuda" and requested_device.index is None:
            requested_device = torch.device("cuda", torch.cuda.current_device())
        self.device = requested_device

        self.max_blocks_per_sequence = (
            max_sequence_length + block_size - 1
        ) // block_size

        pool_shape = (
            num_layers,
            num_gpu_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        )
        self._keys = torch.empty(pool_shape, dtype=dtype, device=self.device)
        self._values = torch.empty_like(self._keys)

        self._block_table = torch.full(
            (max_slots, self.max_blocks_per_sequence),
            fill_value=-1,
            dtype=torch.int32,
            device=self.device,
        )

        self._lengths = [0] * max_slots
        self._active_slots: set[int] = set()
        self._slot_blocks: list[list[int]] = [[] for _ in range(max_slots)]
        self._free_blocks: deque[int] = deque(range(num_gpu_blocks))

        self._next_reservation_id = 0
        self._reservations: dict[int, AppendReservation] = {}
        self._pending_slots: set[int] = set()

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
    def free_block_count(self) -> int:
        return len(self._free_blocks)

    @property
    def allocated_block_count(self) -> int:
        return self.num_gpu_blocks - self.free_block_count

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

    def allocate(self, count: int = 1) -> list[int]:
        self._validate_positive("count", count)

        if count > self.free_slot_count:
            raise KVCacheCapacityError(
                f"requested {count} slots, but only {self.free_slot_count} are free"
            )

        allocated_slots: list[int] = []
        for slot_id in range(self.max_slots):
            if slot_id not in self._active_slots:
                self._active_slots.add(slot_id)
                self._lengths[slot_id] = 0
                self._slot_blocks[slot_id] = []
                self._block_table[slot_id].fill_(-1)
                allocated_slots.append(slot_id)

                if len(allocated_slots) == count:
                    return allocated_slots

        raise AssertionError("free-slot accounting is inconsistent")

    def release(self, slot_ids: Sequence[int]) -> None:
        normalized_slot_ids = self._validate_slot_ids(slot_ids)

        pending = set(normalized_slot_ids) & self._pending_slots
        if pending:
            raise KVCacheError(
                f"cannot release slots with pending append reservations: {sorted(pending)}"
            )

        for slot_id in normalized_slot_ids:
            self._free_blocks.extend(self._slot_blocks[slot_id])
            self._slot_blocks[slot_id] = []
            self._lengths[slot_id] = 0
            self._block_table[slot_id].fill_(-1)
            self._active_slots.remove(slot_id)

    def begin_append(
        self,
        *,
        slot_ids: Sequence[int],
        token_count: int,
    ) -> AppendReservation:
        """
        Reserve logical positions and required blocks without advancing lengths.
        """
        self._validate_positive("token_count", token_count)
        normalized_slot_ids = self._validate_slot_ids(slot_ids)

        pending = set(normalized_slot_ids) & self._pending_slots
        if pending:
            raise KVCacheError(
                f"slots already have pending append reservations: {sorted(pending)}"
            )

        required_new_blocks = self._required_new_blocks(
            normalized_slot_ids,
            token_count,
        )

        if required_new_blocks > self.free_block_count:
            raise KVCacheCapacityError(
                f"append requires {required_new_blocks} free blocks, "
                f"but only {self.free_block_count} remain"
            )

        start_positions = tuple(
            self._lengths[slot_id] for slot_id in normalized_slot_ids
        )
        new_blocks_by_slot: dict[int, tuple[int, ...]] = {}

        for slot_id in normalized_slot_ids:
            new_blocks_by_slot[slot_id] = tuple(
                self._allocate_blocks_for_append(slot_id, token_count)
            )

        reservation = AppendReservation(
            reservation_id=self._next_reservation_id,
            slot_ids=tuple(normalized_slot_ids),
            start_positions=start_positions,
            token_count=token_count,
            new_blocks_by_slot=new_blocks_by_slot,
        )
        self._next_reservation_id += 1
        self._reservations[reservation.reservation_id] = reservation
        self._pending_slots.update(reservation.slot_ids)

        return reservation

    def write_layer(
        self,
        *,
        layer_idx: int,
        reservation: AppendReservation,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Write one transformer layer into the positions held by a reservation."""
        self._validate_layer_idx(layer_idx)
        self._validate_reservation(reservation)
        self._validate_append_tensors(
            keys,
            values,
            expected_batch_size=len(reservation.slot_ids),
            expected_token_count=reservation.token_count,
        )

        if layer_idx in reservation.written_layers:
            raise KVCacheError(
                f"layer {layer_idx} was already written for reservation "
                f"{reservation.reservation_id}"
            )

        for batch_index, (slot_id, start_position) in enumerate(
            zip(reservation.slot_ids, reservation.start_positions)
        ):
            for token_offset in range(reservation.token_count):
                token_position = start_position + token_offset
                logical_block_index = token_position // self.block_size
                offset_in_block = token_position % self.block_size
                physical_block_id = self._slot_blocks[slot_id][logical_block_index]

                self._keys[
                    layer_idx,
                    physical_block_id,
                    offset_in_block,
                ].copy_(keys[batch_index, token_offset])

                self._values[
                    layer_idx,
                    physical_block_id,
                    offset_in_block,
                ].copy_(values[batch_index, token_offset])

        reservation.written_layers.add(layer_idx)

    def commit_append(self, reservation: AppendReservation) -> None:
        """Make all layer writes visible by atomically advancing sequence lengths."""
        self._validate_reservation(reservation)

        required_layers = set(range(self.num_layers))
        if reservation.written_layers != required_layers:
            missing_layers = sorted(required_layers - reservation.written_layers)
            raise KVCacheError(
                f"cannot commit reservation {reservation.reservation_id}: "
                f"all layers must be written, missing {missing_layers}"
            )

        for slot_id, start_position in zip(
            reservation.slot_ids,
            reservation.start_positions,
        ):
            self._lengths[slot_id] = start_position + reservation.token_count

        self._close_reservation(reservation)

    def abort_append(self, reservation: AppendReservation) -> None:
        """
        Roll back only blocks allocated by this reservation.

        Already written data need not be cleared because sequence lengths were
        never advanced, so those positions are not visible to readers.
        """
        self._validate_reservation(reservation)

        for slot_id in reservation.slot_ids:
            new_blocks = reservation.new_blocks_by_slot[slot_id]

            for expected_block_id in reversed(new_blocks):
                actual_block_id = self._slot_blocks[slot_id].pop()
                if actual_block_id != expected_block_id:
                    raise AssertionError("slot block metadata is inconsistent")

                logical_block_index = len(self._slot_blocks[slot_id])
                self._block_table[slot_id, logical_block_index] = -1
                self._free_blocks.appendleft(actual_block_id)

        self._close_reservation(reservation)

    def append(
        self,
        *,
        layer_idx: int,
        slot_ids: Sequence[int],
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """
        Backward-compatible convenience API for a one-layer cache only.

        Multi-layer model code must use begin_append/write_layer/commit_append.
        """
        if self.num_layers != 1:
            raise KVCacheError(
                "append is only valid for num_layers == 1; "
                "use begin_append, write_layer, and commit_append"
            )

        reservation = self.begin_append(
            slot_ids=slot_ids,
            token_count=keys.size(1),
        )

        try:
            self.write_layer(
                layer_idx=layer_idx,
                reservation=reservation,
                keys=keys,
                values=values,
            )
            self.commit_append(reservation)
        except Exception:
            if reservation.reservation_id in self._reservations:
                self.abort_append(reservation)
            raise

    def get_kv(
        self,
        *,
        layer_idx: int,
        slot_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather paged K/V into contiguous tensors for correctness checking.

        A production paged-attention kernel consumes block_table directly rather
        than materializing contiguous historical K/V on every decode step.
        """
        self._validate_layer_idx(layer_idx)
        self._validate_slot_ids([slot_id])

        sequence_length = self._lengths[slot_id]
        gathered_keys = torch.empty(
            (sequence_length, self.num_kv_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        gathered_values = torch.empty_like(gathered_keys)

        copied_tokens = 0
        for physical_block_id in self._slot_blocks[slot_id]:
            token_count = min(self.block_size, sequence_length - copied_tokens)
            if token_count == 0:
                break

            gathered_keys[copied_tokens : copied_tokens + token_count].copy_(
                self._keys[layer_idx, physical_block_id, :token_count]
            )
            gathered_values[copied_tokens : copied_tokens + token_count].copy_(
                self._values[layer_idx, physical_block_id, :token_count]
            )
            copied_tokens += token_count

        if copied_tokens != sequence_length:
            raise AssertionError("block table does not cover the logical sequence")

        return gathered_keys, gathered_values

    def block_table(self, slot_id: int) -> torch.Tensor:
        self._validate_slot_ids([slot_id])
        block_count = len(self._slot_blocks[slot_id])
        return self._block_table[slot_id, :block_count].clone()

    def length(self, slot_id: int) -> int:
        self._validate_slot_ids([slot_id])
        return self._lengths[slot_id]

    def summary(self) -> dict[str, int | float]:
        allocated_token_capacity = self.allocated_block_count * self.block_size
        logical_tokens = sum(self._lengths)

        return {
            "active_slots": self.active_slot_count,
            "free_slots": self.free_slot_count,
            "allocated_blocks": self.allocated_block_count,
            "free_blocks": self.free_block_count,
            "pending_reservations": len(self._reservations),
            "logical_tokens": logical_tokens,
            "block_token_utilization": (
                logical_tokens / allocated_token_capacity
                if allocated_token_capacity > 0
                else 0.0
            ),
            "physical_memory_mib": self.physical_memory_mib,
            "logical_memory_mib": self.logical_memory_bytes() / (1024 * 1024),
        }

    def _required_new_blocks(
        self,
        slot_ids: Sequence[int],
        token_count: int,
    ) -> int:
        required_new_blocks = 0

        for slot_id in slot_ids:
            current_length = self._lengths[slot_id]
            new_length = current_length + token_count

            if new_length > self.max_sequence_length:
                raise KVCacheCapacityError(
                    f"slot {slot_id} append exceeds capacity: "
                    f"{current_length} + {token_count} > {self.max_sequence_length}"
                )

            required_block_count = (
                new_length + self.block_size - 1
            ) // self.block_size
            required_new_blocks += (
                required_block_count - len(self._slot_blocks[slot_id])
            )

        return required_new_blocks

    def _allocate_blocks_for_append(
        self,
        slot_id: int,
        token_count: int,
    ) -> list[int]:
        required_block_count = (
            self._lengths[slot_id] + token_count + self.block_size - 1
        ) // self.block_size

        new_blocks: list[int] = []

        while len(self._slot_blocks[slot_id]) < required_block_count:
            physical_block_id = self._free_blocks.popleft()
            logical_block_index = len(self._slot_blocks[slot_id])

            self._slot_blocks[slot_id].append(physical_block_id)
            self._block_table[slot_id, logical_block_index] = physical_block_id
            new_blocks.append(physical_block_id)

        return new_blocks

    def _close_reservation(self, reservation: AppendReservation) -> None:
        del self._reservations[reservation.reservation_id]
        self._pending_slots.difference_update(reservation.slot_ids)

    def _validate_reservation(self, reservation: AppendReservation) -> None:
        active_reservation = self._reservations.get(reservation.reservation_id)
        if active_reservation is not reservation:
            raise KVCacheError(
                f"reservation {reservation.reservation_id} is not active"
            )

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
        *,
        expected_batch_size: int,
        expected_token_count: int,
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
                f"batch size {batch_size} does not match "
                f"{expected_batch_size} reservation slots"
            )

        if token_count != expected_token_count:
            raise ValueError(
                f"token count {token_count} does not match "
                f"{expected_token_count} reserved tokens"
            )

        if num_kv_heads != self.num_kv_heads or head_dim != self.head_dim:
            raise ValueError(
                "KV tensor head shape does not match cache configuration: "
                f"expected ({self.num_kv_heads}, {self.head_dim}), "
                f"got ({num_kv_heads}, {head_dim})"
            )

        for name, tensor in (("keys", keys), ("values", values)):
            if tensor.dtype != self.dtype:
                raise ValueError(
                    f"{name} must have dtype {self.dtype}, got {tensor.dtype}"
                )
            if tensor.device != self.device:
                raise ValueError(
                    f"{name} must be on {self.device}, got {tensor.device}"
                )