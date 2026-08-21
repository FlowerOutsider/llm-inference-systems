from serving.control_plane.continuous_batch_scheduler import (
    ContinuousBatchScheduler,
    SchedulerConfig,
)
from serving.control_plane.request_state import (
    InferenceRequest,
    RequestPhase,
)

import pytest

def make_request(
    request_id: str,
    prompt_length: int,
    max_new_tokens: int = 4,
) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        prompt_token_ids=tuple(range(prompt_length)),
        max_new_tokens=max_new_tokens,
    )


def test_decode_has_priority_but_prefill_can_fill_remaining_token_budget() -> None:
    scheduler = ContinuousBatchScheduler(
        SchedulerConfig(
            max_num_seqs=2,
            max_num_batched_tokens=5,
            prefill_chunk_size=4,
        )
    )

    decoding = make_request("decoding", prompt_length=3)
    waiting = make_request("waiting", prompt_length=10)

    scheduler.submit(decoding)
    scheduler.submit(waiting)

    scheduler.mark_prefill_executed(
        request_id="decoding",
        token_count=3,
    )
    assert decoding.phase is RequestPhase.DECODE

    plan = scheduler.schedule()

    assert [item.request_id for item in plan.decode] == ["decoding"]
    assert [(item.request_id, item.token_start, item.token_end) for item in plan.prefill] == [
        ("waiting", 0, 4),
    ]
    assert plan.total_scheduled_tokens == 5


def test_prefill_is_chunked_and_request_enters_decode_only_after_full_prompt() -> None:
    scheduler = ContinuousBatchScheduler(
        SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=6,
            prefill_chunk_size=4,
        )
    )
    request = make_request("long-prompt", prompt_length=10)
    scheduler.submit(request)

    first_plan = scheduler.schedule()
    assert [(item.token_start, item.token_end) for item in first_plan.prefill] == [
        (0, 4)
    ]
    scheduler.mark_prefill_executed("long-prompt", token_count=4)
    assert request.phase is RequestPhase.PREFILL
    assert request.prefill_offset == 4

    second_plan = scheduler.schedule()
    assert [(item.token_start, item.token_end) for item in second_plan.prefill] == [
        (4, 8)
    ]
    scheduler.mark_prefill_executed("long-prompt", token_count=4)

    third_plan = scheduler.schedule()
    assert [(item.token_start, item.token_end) for item in third_plan.prefill] == [
        (8, 10)
    ]
    scheduler.mark_prefill_executed("long-prompt", token_count=2)

    assert request.phase is RequestPhase.DECODE
    assert request.prefill_offset == 10


def test_active_decode_uses_sequence_capacity_before_waiting_prefill() -> None:
    scheduler = ContinuousBatchScheduler(
        SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=8,
            prefill_chunk_size=8,
        )
    )

    decoding = make_request("decoding", prompt_length=2)
    waiting = make_request("waiting", prompt_length=8)

    scheduler.submit(decoding)
    scheduler.submit(waiting)
    scheduler.mark_prefill_executed("decoding", token_count=2)

    plan = scheduler.schedule()

    assert [item.request_id for item in plan.decode] == ["decoding"]
    assert plan.prefill == ()
    assert plan.total_scheduled_tokens == 1


def test_cancelled_and_finished_requests_are_never_scheduled() -> None:
    scheduler = ContinuousBatchScheduler(
        SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=16,
            prefill_chunk_size=8,
        )
    )

    cancelled = make_request("cancelled", prompt_length=4)
    finished = make_request("finished", prompt_length=2, max_new_tokens=1)

    scheduler.submit(cancelled)
    scheduler.submit(finished)

    scheduler.cancel("cancelled")
    scheduler.mark_prefill_executed("finished", token_count=2)
    scheduler.mark_decode_executed("finished", token_id=42)

    assert cancelled.phase is RequestPhase.CANCELLED
    assert finished.phase is RequestPhase.FINISHED

    plan = scheduler.schedule()

    assert plan.decode == ()
    assert plan.prefill == ()
    assert plan.total_scheduled_tokens == 0


def test_decode_completion_respects_max_new_tokens() -> None:
    scheduler = ContinuousBatchScheduler(
        SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=4,
            prefill_chunk_size=4,
        )
    )
    request = make_request("generate-two", prompt_length=1, max_new_tokens=2)
    scheduler.submit(request)

    scheduler.mark_prefill_executed("generate-two", token_count=1)

    first_plan = scheduler.schedule()
    assert [item.request_id for item in first_plan.decode] == ["generate-two"]
    scheduler.mark_decode_executed("generate-two", token_id=101)

    second_plan = scheduler.schedule()
    assert [item.request_id for item in second_plan.decode] == ["generate-two"]
    scheduler.mark_decode_executed("generate-two", token_id=102)

    assert request.phase is RequestPhase.FINISHED
    assert request.generated_token_ids == [101, 102]
    assert scheduler.schedule().decode == ()

def test_failed_request_is_never_scheduled_and_keeps_failure_reason() -> None:
    scheduler = ContinuousBatchScheduler(
        SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=4,
            prefill_chunk_size=4,
        )
    )
    request = make_request("backend-failure", prompt_length=2)
    scheduler.submit(request)

    scheduler.fail("backend-failure", reason="vLLM worker timeout")

    assert request.phase is RequestPhase.FAILED
    assert request.failure_reason == "vLLM worker timeout"
    assert scheduler.schedule().total_scheduled_tokens == 0


def test_remove_requires_terminal_request() -> None:
    scheduler = ContinuousBatchScheduler(
        SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=4,
            prefill_chunk_size=4,
        )
    )
    request = make_request("remove-me", prompt_length=1)
    scheduler.submit(request)

    with pytest.raises(ValueError, match="non-terminal"):
        scheduler.remove("remove-me")

    scheduler.cancel("remove-me")
    scheduler.remove("remove-me")

    with pytest.raises(KeyError, match="remove-me"):
        scheduler.get_request("remove-me")