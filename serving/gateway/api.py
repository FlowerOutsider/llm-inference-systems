from __future__ import annotations

import json
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from serving.data_plane.vllm_worker_adapter import (
    ChatMessage,
    StreamingChatCompletionEvent,
    VLLMWorkerError,
)
from serving.gateway.gateway_service import (
    GatewayChatRequest,
    GatewayNoAvailableWorkerError,
    GatewayService,
)


class ChatMessageBody(BaseModel):
    role: str = Field(min_length=1)
    content: str = Field(min_length=1)


class ChatCompletionBody(BaseModel):
    model: str = Field(min_length=1)
    messages: list[ChatMessageBody] = Field(min_length=1)
    max_tokens: int = Field(default=64, gt=0)
    stream: bool = True


def create_app(*, service: GatewayService) -> FastAPI:
    app = FastAPI(
        title="LLM Inference MaaS Gateway",
        version="0.1.0",
    )

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionBody,
    ) -> StreamingResponse:
        if not body.stream:
            raise HTTPException(
                status_code=400,
                detail="only stream=true is supported",
            )

        request = GatewayChatRequest(
            request_id=uuid4().hex,
            model_id=body.model,
            messages=tuple(
                ChatMessage(
                    role=message.role,
                    content=message.content,
                )
                for message in body.messages
            ),
            max_tokens=body.max_tokens,
            prefix_candidate_worker_ids=frozenset(),
        )

        try:
            event_stream = await service.prepare_stream_chat_completion(
                request
            )
        except GatewayNoAvailableWorkerError as exc:
            raise HTTPException(
                status_code=503,
                detail=str(exc),
            ) from exc

        return StreamingResponse(
            _encode_sse_events(event_stream),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


async def _encode_sse_events(
    event_stream: AsyncIterator[StreamingChatCompletionEvent],
) -> AsyncIterator[str]:
    try:
        async for event in event_stream:
            payload = {
                "id": event.request_id,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": event.delta_content,
                        },
                        "finish_reason": event.finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n"
    except VLLMWorkerError as exc:
        payload = {
            "error": {
                "type": "upstream_error",
                "message": str(exc),
            }
        }
        yield f"data: {json.dumps(payload)}\n\n"
    finally:
        yield "data: [DONE]\n\n"