from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class RequestPhase(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RequestStateError(RuntimeError):
    """Raised when a request attempts an invalid lifecycle transition."""


@dataclass
class InferenceRequest:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    phase: RequestPhase = RequestPhase.WAITING
    prefill_offset: int = 0
    generated_token_ids: list[int] = field(default_factory=list)
    failure_reason: str | None = None


    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")

        self.prompt_token_ids = tuple(self.prompt_token_ids)

        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.prefill_offset < 0 or self.prefill_offset > self.prompt_length:
            raise ValueError("prefill_offset is outside the prompt range")

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def remaining_prefill_tokens(self) -> int:
        return self.prompt_length - self.prefill_offset

    @property
    def is_terminal(self) -> bool:
        return self.phase in {
            RequestPhase.FINISHED,
            RequestPhase.CANCELLED,
            RequestPhase.FAILED,
        }

    def advance_prefill(self, token_count: int) -> None:
        if self.phase not in {RequestPhase.WAITING, RequestPhase.PREFILL}:
            raise RequestStateError(
                f"cannot execute prefill while request is {self.phase.value}"
            )
        if token_count <= 0:
            raise ValueError("prefill token_count must be positive")
        if token_count > self.remaining_prefill_tokens:
            raise RequestStateError(
                "prefill token_count exceeds remaining prompt tokens"
            )

        self.phase = RequestPhase.PREFILL
        self.prefill_offset += token_count

        if self.prefill_offset == self.prompt_length:
            self.phase = RequestPhase.DECODE

    def append_generated_token(self, token_id: int) -> None:
        if self.phase is not RequestPhase.DECODE:
            raise RequestStateError(
                f"cannot decode while request is {self.phase.value}"
            )
        if len(self.generated_token_ids) >= self.max_new_tokens:
            raise RequestStateError("request has already generated max_new_tokens")

        self.generated_token_ids.append(token_id)

        if len(self.generated_token_ids) == self.max_new_tokens:
            self.phase = RequestPhase.FINISHED

    def cancel(self) -> None:
        if self.is_terminal:
            return

        self.phase = RequestPhase.CANCELLED

    def fail(self, reason: str) -> None:
        if self.is_terminal:
            return

        normalized_reason = reason.strip()
        if not normalized_reason:
            raise ValueError("failure reason must not be empty")

        self.failure_reason = normalized_reason
        self.phase = RequestPhase.FAILED