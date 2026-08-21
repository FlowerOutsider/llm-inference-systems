from __future__ import annotations

from serving.control_plane.continuous_batch_scheduler import (
    ContinuousBatchScheduler,
    SchedulePlan,
)
from serving.control_plane.prefix_index import PrefixScope
from serving.control_plane.request_lifecycle import (
    RequestAdmission,
    RequestLifecycleManager,
)
from serving.control_plane.request_state import InferenceRequest, RequestPhase


class InferenceRuntimeError(RuntimeError):
    """推理运行时执行顺序或确认流程不合法。"""


class InferenceRuntime:
    """
    控制平面运行时。

    Scheduler 只负责产生调度计划；Runtime 负责把计划交给后端执行，
    并且仅在收到执行确认后推进请求状态。
    """

    def __init__(
        self,
        *,
        lifecycle: RequestLifecycleManager,
        scheduler: ContinuousBatchScheduler,
    ) -> None:
        self._lifecycle = lifecycle
        self._scheduler = scheduler
        self._pending_prefill: dict[str, int] = {}
        self._pending_decode: set[str] = set()

    @property
    def pending_work_item_count(self) -> int:
        return len(self._pending_prefill) + len(self._pending_decode)

    def submit(
        self,
        request: InferenceRequest,
        *,
        prefix_scope: PrefixScope | None = None,
    ) -> RequestAdmission:
        return self._lifecycle.submit(request, prefix_scope=prefix_scope)

    def schedule_once(self) -> SchedulePlan:
        if self.pending_work_item_count:
            raise InferenceRuntimeError(
                "cannot schedule while previous work is unacknowledged"
            )

        plan = self._scheduler.schedule()

        pending_prefill = {
            item.request_id: item.token_count
            for item in plan.prefill
        }
        pending_decode = {
            item.request_id
            for item in plan.decode
        }

        overlap = set(pending_prefill).intersection(pending_decode)
        if overlap:
            raise InferenceRuntimeError(
                f"request appears in both prefill and decode work: {sorted(overlap)}"
            )

        self._pending_prefill = pending_prefill
        self._pending_decode = pending_decode
        return plan

    def acknowledge_prefill(
        self,
        request_id: str,
        *,
        token_count: int,
    ) -> None:
        expected_token_count = self._pending_prefill.get(request_id)
        if expected_token_count is None:
            raise InferenceRuntimeError(
                f"no pending prefill work for request {request_id!r}"
            )
        if token_count != expected_token_count:
            raise InferenceRuntimeError(
                f"prefill acknowledgement for {request_id!r} expected "
                f"{expected_token_count} tokens, got {token_count}"
            )

        self._scheduler.mark_prefill_executed(
            request_id,
            token_count=token_count,
        )
        del self._pending_prefill[request_id]

    def acknowledge_decode(
        self,
        request_id: str,
        *,
        token_id: int,
    ) -> None:
        if request_id not in self._pending_decode:
            raise InferenceRuntimeError(
                f"no pending decode work for request {request_id!r}"
            )

        self._scheduler.mark_decode_executed(
            request_id,
            token_id=token_id,
        )

        request = self._scheduler.get_request(request_id)
        if request.phase is RequestPhase.FINISHED:
            self._lifecycle.finalize(request_id)

        self._pending_decode.remove(request_id)

    def cancel(self, request_id: str) -> None:
        self._lifecycle.cancel(request_id)
        self._discard_pending_work(request_id)

    def fail(self, request_id: str, *, reason: str) -> None:
        self._lifecycle.fail(request_id, reason=reason)
        self._discard_pending_work(request_id)

    def _discard_pending_work(self, request_id: str) -> None:
        self._pending_prefill.pop(request_id, None)
        self._pending_decode.discard(request_id)