from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from serving.control_plane.request_state import (
    InferenceRequest,
    RequestPhase,
)


@dataclass(frozen=True)
class SchedulerConfig:
    max_num_seqs: int
    max_num_batched_tokens: int
    prefill_chunk_size: int

    def __post_init__(self) -> None:
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")


@dataclass(frozen=True)
class DecodeWorkItem:
    request_id: str


@dataclass(frozen=True)
class PrefillWorkItem:
    request_id: str
    token_start: int
    token_end: int

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start


@dataclass(frozen=True)
class SchedulePlan:
    decode: tuple[DecodeWorkItem, ...]
    prefill: tuple[PrefillWorkItem, ...]
    total_scheduled_tokens: int


class ContinuousBatchScheduler:
    """Creates deterministic continuous-batching plans.

    The scheduler never advances request state itself. The model execution
    layer must acknowledge completed work through mark_prefill_executed()
    and mark_decode_executed().
    """

    def __init__(self, config: SchedulerConfig) -> None:
        self._config = config
        self._requests: dict[str, InferenceRequest] = {}

    def submit(self, request: InferenceRequest) -> None:
        if request.request_id in self._requests:
            raise ValueError(f"duplicate request_id: {request.request_id}")
        if request.is_terminal:
            raise ValueError("cannot submit a terminal request")

        self._requests[request.request_id] = request

    def cancel(self, request_id: str) -> None:
        self._get_request(request_id).cancel()

    def schedule(self) -> SchedulePlan:
        remaining_token_budget = self._config.max_num_batched_tokens
        remaining_sequence_budget = self._config.max_num_seqs

        decode_items: list[DecodeWorkItem] = []
        prefill_items: list[PrefillWorkItem] = []

        # Decode receives priority because delaying an active sequence worsens TPOT.
        for request in self._requests.values():
            if (
                request.phase is not RequestPhase.DECODE
                or remaining_token_budget < 1
                or remaining_sequence_budget < 1
            ):
                continue

            decode_items.append(DecodeWorkItem(request_id=request.request_id))
            remaining_token_budget -= 1
            remaining_sequence_budget -= 1

        # Use remaining capacity for prompt processing. One request receives at
        # most one prefill chunk in a scheduling tick for fairness.
        for request in self._requests.values():
            if (
                request.phase
                not in {RequestPhase.WAITING, RequestPhase.PREFILL}
                or remaining_token_budget <= 0
                or remaining_sequence_budget <= 0
            ):
                continue

            token_count = min(
                request.remaining_prefill_tokens,
                self._config.prefill_chunk_size,
                remaining_token_budget,
            )
            if token_count <= 0:
                continue

            token_start = request.prefill_offset
            token_end = token_start + token_count

            prefill_items.append(
                PrefillWorkItem(
                    request_id=request.request_id,
                    token_start=token_start,
                    token_end=token_end,
                )
            )
            remaining_token_budget -= token_count
            remaining_sequence_budget -= 1

        total_scheduled_tokens = (
            len(decode_items)
            + sum(item.token_count for item in prefill_items)
        )

        return SchedulePlan(
            decode=tuple(decode_items),
            prefill=tuple(prefill_items),
            total_scheduled_tokens=total_scheduled_tokens,
        )

    def mark_prefill_executed(
        self,
        request_id: str,
        token_count: int,
    ) -> None:
        self._get_request(request_id).advance_prefill(token_count)

    def mark_decode_executed(
        self,
        request_id: str,
        token_id: int,
    ) -> None:
        self._get_request(request_id).append_generated_token(token_id)

    def get_request(self, request_id: str) -> InferenceRequest:
        return self._get_request(request_id)

    def active_request_ids(self) -> tuple[str, ...]:
        return tuple(
            request.request_id
            for request in self._requests.values()
            if not request.is_terminal
        )

    def _get_request(self, request_id: str) -> InferenceRequest:
        try:
            return self._requests[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown request_id: {request_id}") from exc