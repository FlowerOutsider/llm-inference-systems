import asyncio
import json
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from serving.data_plane.vllm_worker_adapter import (
    StreamingChatCompletionEvent,
    VLLMWorkerTransportError,
)
from serving.gateway.api import create_app
from serving.gateway.gateway_service import (
    GatewayChatRequest,
    GatewayNoAvailableWorkerError,
)


def run(coroutine):
    return asyncio.run(coroutine)


@dataclass
class FakeGatewayService:
    events: list[StreamingChatCompletionEvent] = field(
        default_factory=list
    )
    preparation_error: Exception | None = None
    streaming_error: Exception | None = None
    requests: list[GatewayChatRequest] = field(default_factory=list)

    async def prepare_stream_chat_completion(
        self,
        request: GatewayChatRequest,
    ) -> AsyncIterator[StreamingChatCompletionEvent]:
        self.requests.append(request)

        if self.preparation_error is not None:
            raise self.preparation_error

        async def stream() -> AsyncIterator[StreamingChatCompletionEvent]:
            for event in self.events:
                yield event

            if self.streaming_error is not None:
                raise self.streaming_error

        return stream()


async def post_json(
    app,
    payload: dict[str, object],
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://gateway.test",
    ) as client:
        return await client.post(
            "/v1/chat/completions",
            json=payload,
        )


def make_payload(*, stream: bool = True) -> dict[str, object]:
    return {
        "model": "qwen2.5-0.5b",
        "messages": [
            {
                "role": "user",
                "content": "hello",
            }
        ],
        "max_tokens": 16,
        "stream": stream,
    }


def sse_payloads(body: str) -> list[str]:
    return [
        line.removeprefix("data: ").strip()
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def test_chat_completions_returns_openai_style_sse() -> None:
    service = FakeGatewayService(
        events=[
            StreamingChatCompletionEvent(
                request_id="chatcmpl-1",
                delta_content="hello",
                finish_reason=None,
            ),
            StreamingChatCompletionEvent(
                request_id="chatcmpl-1",
                delta_content=" world",
                finish_reason="stop",
            ),
        ]
    )
    response = run(post_json(create_app(service=service), make_payload()))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "text/event-stream"
    )

    payloads = sse_payloads(response.text)
    first_chunk = json.loads(payloads[0])
    second_chunk = json.loads(payloads[1])

    assert first_chunk == {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "hello"},
                "finish_reason": None,
            }
        ],
    }
    assert second_chunk["choices"][0]["delta"]["content"] == " world"
    assert second_chunk["choices"][0]["finish_reason"] == "stop"
    assert payloads[-1] == "[DONE]"

    assert len(service.requests) == 1
    assert service.requests[0].model_id == "qwen2.5-0.5b"
    assert service.requests[0].max_tokens == 16


def test_chat_completions_returns_503_before_stream_when_no_worker() -> None:
    service = FakeGatewayService(
        preparation_error=GatewayNoAvailableWorkerError(
            "no eligible worker for model"
        )
    )
    response = run(post_json(create_app(service=service), make_payload()))

    assert response.status_code == 503
    assert response.json()["detail"] == "no eligible worker for model"


def test_chat_completions_rejects_non_streaming_request() -> None:
    service = FakeGatewayService()
    response = run(
        post_json(
            create_app(service=service),
            make_payload(stream=False),
        )
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "only stream=true is supported"
    )
    assert service.requests == []


def test_chat_completions_emits_sse_error_after_stream_starts() -> None:
    service = FakeGatewayService(
        events=[
            StreamingChatCompletionEvent(
                request_id="chatcmpl-1",
                delta_content="partial",
                finish_reason=None,
            )
        ],
        streaming_error=VLLMWorkerTransportError(
            "vLLM transport request failed: read timeout"
        ),
    )
    response = run(post_json(create_app(service=service), make_payload()))

    assert response.status_code == 200

    payloads = sse_payloads(response.text)
    error_event = json.loads(payloads[-2])

    assert error_event["error"]["type"] == "upstream_error"
    assert "read timeout" in error_event["error"]["message"]
    assert payloads[-1] == "[DONE]"