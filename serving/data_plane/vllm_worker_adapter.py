from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx


class VLLMWorkerError(RuntimeError):
    """vLLM Worker 调用失败的基类。"""


class VLLMWorkerHTTPError(VLLMWorkerError):
    def __init__(self, *, status_code: int, response_text: str) -> None:
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(
            f"vLLM returned HTTP {status_code}: {response_text}"
        )


class VLLMWorkerTransportError(VLLMWorkerError):
    """连接、读取、DNS 或超时等网络层错误。"""


class VLLMWorkerProtocolError(VLLMWorkerError):
    """vLLM 返回的 SSE 或 JSON 不符合约定。"""


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ValueError("message role must not be empty")
        if not self.content:
            raise ValueError("message content must not be empty")


@dataclass(frozen=True)
class StreamingChatCompletionEvent:
    request_id: str
    delta_content: str
    finish_reason: str | None


class VLLMOpenAIWorker:
    """
    vLLM OpenAI 兼容接口的流式客户端。

    该类负责请求编码、SSE 解析和错误归类；不负责 vLLM 内部的
    Continuous Batching、KV Cache 或 token 级调度。
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        normalized_base_url = base_url.rstrip("/")

        if not normalized_base_url:
            raise ValueError("base_url must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self._base_url = normalized_base_url
        self._model = model
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stream_chat_completion(
        self,
        *,
        messages: list[ChatMessage],
        max_tokens: int,
    ) -> AsyncIterator[StreamingChatCompletionEvent]:
        if not messages:
            raise ValueError("messages must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        payload = {
            "model": self._model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "max_tokens": max_tokens,
            "stream": True,
        }

        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                json=payload,
            ) as response:
                if response.is_error:
                    response_text = (
                        await response.aread()
                    ).decode("utf-8", errors="replace")
                    raise VLLMWorkerHTTPError(
                        status_code=response.status_code,
                        response_text=response_text,
                    )

                saw_done = False

                async for line in response.aiter_lines():
                    event_data = self._extract_sse_data(line)
                    if event_data is None:
                        continue

                    if event_data == "[DONE]":
                        saw_done = True
                        break

                    event = self._parse_event(event_data)
                    if event is not None:
                        yield event

                if not saw_done:
                    raise VLLMWorkerProtocolError(
                        "vLLM stream ended without a [DONE] event"
                    )
        except VLLMWorkerError:
            raise
        except httpx.RequestError as exc:
            raise VLLMWorkerTransportError(
                f"vLLM transport request failed: {exc}"
            ) from exc

    @staticmethod
    def _extract_sse_data(line: str) -> str | None:
        stripped_line = line.strip()

        if not stripped_line:
            return None
        if not stripped_line.startswith("data:"):
            return None

        return stripped_line.removeprefix("data:").strip()

    @staticmethod
    def _parse_event(
        event_data: str,
    ) -> StreamingChatCompletionEvent | None:
        try:
            payload: Any = json.loads(event_data)
        except json.JSONDecodeError as exc:
            raise VLLMWorkerProtocolError(
                f"invalid JSON in vLLM SSE event: {event_data!r}"
            ) from exc

        if not isinstance(payload, dict):
            raise VLLMWorkerProtocolError(
                "vLLM SSE payload must be a JSON object"
            )

        request_id = payload.get("id")
        if not isinstance(request_id, str) or not request_id:
            raise VLLMWorkerProtocolError(
                "vLLM SSE payload is missing a non-empty id"
            )

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise VLLMWorkerProtocolError(
                "vLLM SSE payload is missing non-empty choices"
            )

        choice = choices[0]
        if not isinstance(choice, dict):
            raise VLLMWorkerProtocolError(
                "vLLM SSE choice must be a JSON object"
            )

        delta = choice.get("delta", {})
        if not isinstance(delta, dict):
            raise VLLMWorkerProtocolError(
                "vLLM SSE delta must be a JSON object"
            )

        content = delta.get("content")
        finish_reason = choice.get("finish_reason")

        if content is not None and not isinstance(content, str):
            raise VLLMWorkerProtocolError(
                "vLLM SSE delta content must be a string"
            )
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise VLLMWorkerProtocolError(
                "vLLM SSE finish_reason must be a string or null"
            )

        if content is None and finish_reason is None:
            return None

        return StreamingChatCompletionEvent(
            request_id=request_id,
            delta_content=content or "",
            finish_reason=finish_reason,
        )