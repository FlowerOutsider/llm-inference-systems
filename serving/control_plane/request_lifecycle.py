from __future__ import annotations

from dataclasses import dataclass

from serving.control_plane.continuous_batch_scheduler import (
    ContinuousBatchScheduler,
)
from serving.control_plane.prefix_cache_manager import PrefixCacheManager
from serving.control_plane.prefix_index import PrefixScope
from serving.control_plane.request_state import (
    InferenceRequest,
    RequestPhase,
)
from serving.data_plane.paged_kv_cache import KVCacheError, PagedKVCache


class RequestAdmissionError(RuntimeError):
    """Raised when a request cannot obtain the resources needed to run."""


class RequestLifecycleError(RuntimeError):
    """Raised when request resource ownership is used incorrectly."""


@dataclass(frozen=True)
class RequestAdmission:
    slot_id: int
    reused_prefix_tokens: int


class RequestLifecycleManager:
    """Owns active-request KV slots across submit, cancel, and finish.

    PrefixCacheManager owns cache-source slots. This manager owns only slots
    belonging to active inference requests.
    """

    def __init__(
        self,
        *,
        cache: PagedKVCache,
        scheduler: ContinuousBatchScheduler,
        prefix_cache_manager: PrefixCacheManager | None = None,
    ) -> None:
        self._cache = cache
        self._scheduler = scheduler
        self._prefix_cache_manager = prefix_cache_manager
        self._active_slots: dict[str, int] = {}

    def submit(
        self,
        request: InferenceRequest,
        *,
        prefix_scope: PrefixScope | None = None,
    ) -> RequestAdmission:
        if request.request_id in self._active_slots:
            raise RequestAdmissionError(
                f"request is already active: {request.request_id}"
            )
        if request.is_terminal:
            raise RequestAdmissionError(
                f"cannot submit terminal request: {request.request_id}"
            )
        if prefix_scope is not None and self._prefix_cache_manager is None:
            raise RequestAdmissionError(
                "prefix_scope was provided but no PrefixCacheManager is configured"
            )

        try:
            slot_id = self._cache.allocate(1)[0]
        except KVCacheError as exc:
            raise RequestAdmissionError(
                f"unable to allocate KV slot for request {request.request_id}: {exc}"
            ) from exc

        submitted_to_scheduler = False

        try:
            self._scheduler.submit(request)
            submitted_to_scheduler = True

            reused_prefix_tokens = 0

            if self._prefix_cache_manager is not None and prefix_scope is not None:
                match = self._prefix_cache_manager.attach_longest_prefix(
                    scope=prefix_scope,
                    token_ids=request.prompt_token_ids,
                    target_slot=slot_id,
                )

                if match is not None:
                    reused_prefix_tokens = match.prefix_length
                    self._scheduler.mark_prefill_executed(
                        request.request_id,
                        token_count=reused_prefix_tokens,
                    )

            self._active_slots[request.request_id] = slot_id

            return RequestAdmission(
                slot_id=slot_id,
                reused_prefix_tokens=reused_prefix_tokens,
            )

        except Exception:
            if submitted_to_scheduler:
                self._scheduler.cancel(request.request_id)

            self._cache.release([slot_id])
            raise

    def cancel(self, request_id: str) -> None:
        self._require_active_slot(request_id)

        self._scheduler.cancel(request_id)
        self._release_slot(request_id)

    def finalize(self, request_id: str) -> None:
        self._require_active_slot(request_id)

        request = self._scheduler.get_request(request_id)
        if request.phase is not RequestPhase.FINISHED:
            raise RequestLifecycleError(
                f"cannot finalize request {request_id} while it is "
                f"{request.phase.value}"
            )

        self._release_slot(request_id)

    def slot_for(self, request_id: str) -> int:
        return self._require_active_slot(request_id)

    def active_request_ids(self) -> tuple[str, ...]:
        return tuple(self._active_slots)

    def _release_slot(self, request_id: str) -> None:
        slot_id = self._active_slots.pop(request_id)
        self._cache.release([slot_id])

    def _require_active_slot(self, request_id: str) -> int:
        try:
            return self._active_slots[request_id]
        except KeyError as exc:
            raise KeyError(f"request has no active KV slot: {request_id}") from exc