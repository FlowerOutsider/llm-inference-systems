import asyncio
import json

import httpx
import pytest

from serving.data_plane.vllm_worker_adapter import (
    ChatMessage,
    VLLMOpenAIWorker,
    VLLMWorkerHTTPError,
    VLLMWorkerProtocolError,
    VLLMWorkerTransportError,
)


def run(coroutine):
    return asyncio.run(coroutine)


def make_worker(handler) -> VLLMOpenAIWorker:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://vllm.test",
    )
    return VLLMOpenAIWorker(
        base_url="http://vllm.test",
        model="qwen-test",
        client=client,
    )


async def collect_events(
    worker: VLLMOpenAIWorker,
) -> list:
    return [
        event
        async for event in worker.stream_chat_completion(
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=16,
        )
    ]


def test_stream_chat_completion_sends_expected_payload_and_parses_sse() -> None:
    captured_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload

        assert request.method == "POST"
        assert request.url == "http://vllm.test/v1/chat/completions"

        captured_payload = json.loads(request.content.decode("utf-8"))

        body = "\n".join(
            [
                'data: {"id":"chatcmpl-1","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}',
                "",
                'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"hello"},"finish_reason":null}]}',
                "",
                'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}',
                "",
                "data: [DONE]",
                "",
            ]
        )
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=body.encode("utf-8"),
        )

    worker = make_worker(handler)

    try:
        events = run(collect_events(worker))
    finally:
        run(worker.aclose())

    assert captured_payload == {
        "model": "qwen-test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16,
        "stream": True,
    }
    assert [event.delta_content for event in events] == ["hello", " world"]
    assert [event.finish_reason for event in events] == [None, "stop"]
    assert all(event.request_id == "chatcmpl-1" for event in events)


def test_stream_chat_completion_maps_http_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=503,
            text="worker overloaded",
        )

    worker = make_worker(handler)

    try:
        with pytest.raises(VLLMWorkerHTTPError) as exc_info:
            run(collect_events(worker))
    finally:
        run(worker.aclose())

    assert exc_info.value.status_code == 503
    assert exc_info.value.response_text == "worker overloaded"


def test_stream_chat_completion_rejects_malformed_sse_json() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=b"data: this-is-not-json\n\n",
        )

    worker = make_worker(handler)

    try:
        with pytest.raises(VLLMWorkerProtocolError, match="invalid JSON"):
            run(collect_events(worker))
    finally:
        run(worker.aclose())


def test_stream_chat_completion_rejects_missing_choices() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=b'data: {"id":"chatcmpl-1"}\n\n',
        )

    worker = make_worker(handler)

    try:
        with pytest.raises(VLLMWorkerProtocolError, match="choices"):
            run(collect_events(worker))
    finally:
        run(worker.aclose())


def test_stream_chat_completion_maps_transport_error() -> None:
    class FailingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(
            self,
            request: httpx.Request,
        ) -> httpx.Response:
            raise httpx.ReadTimeout(
                "vLLM response timed out",
                request=request,
            )

    client = httpx.AsyncClient(
        transport=FailingTransport(),
        base_url="http://vllm.test",
    )
    worker = VLLMOpenAIWorker(
        base_url="http://vllm.test",
        model="qwen-test",
        client=client,
    )

    try:
        with pytest.raises(VLLMWorkerTransportError, match="timed out"):
            run(collect_events(worker))
    finally:
        run(worker.aclose())