from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from minirag.domain.ports import InferenceEngine, InferenceError
from minirag.observability import get_client, observe


class OpenRouterEngine(InferenceEngine):
    """LLM inference engine backed by the OpenRouter API, with reasoning support."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    @dataclass(frozen=True)
    class _Usage:
        input_tokens: int | None
        output_tokens: int | None

        def as_langfuse(self) -> dict[str, int] | None:
            usage = {
                key: value
                for key, value in {
                    "input": self.input_tokens,
                    "output": self.output_tokens,
                }.items()
                if value is not None
            }
            return usage or None

    @dataclass(frozen=True)
    class _Result:
        message: dict[str, Any]
        usage: OpenRouterEngine._Usage

    def __init__(
        self,
        model: str,
        api_key: str,
        timeout: float = 60.0,
    ):
        self.model = model
        self.api_key = api_key
        if not self.api_key or not self.model:
            raise RuntimeError(
                "OpenRouter API key is required. Pass api_key=... or set OPENROUTER_API_KEY."
            )
        if timeout <= 0:
            raise ValueError("OpenRouter inference timeout must be greater than zero")
        self._timeout = timeout

    def _prepare_messages(
        self, messages: str | list[dict[str, Any]], last_response: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        if isinstance(messages, str):
            prepared_messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            prepared_messages = [dict(message) for message in messages]
        else:
            raise TypeError("messages must be a string or a list of message objects")

        if last_response is not None:
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": last_response.get("content"),
            }
            reasoning = last_response.get("reasoning_details")
            if reasoning is not None:
                assistant_msg["reasoning_details"] = reasoning
            prepared_messages = [assistant_msg, *prepared_messages]

        return prepared_messages

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        reasoning: bool,
        schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "reasoning": {"enabled": reasoning},
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "miniRAG",
                    "strict": True,
                    "schema": schema,
                },
            }
        return payload

    def _request_completion(self, payload: dict[str, Any]) -> object:
        try:
            response = requests.post(
                self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise InferenceError(f"LLM inference failed: {exc}") from exc

        try:
            return response.json()
        except ValueError as exc:
            raise InferenceError("Unexpected response format: invalid JSON") from exc

    def _parse_response(self, body: object) -> _Result:
        if not isinstance(body, dict):
            raise InferenceError("Unexpected response format: expected an object")
        if "error" in body:
            raise InferenceError(f"OpenRouter API error: {body['error']}")

        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError(
                "Unexpected response format: missing assistant message"
            ) from exc
        if not isinstance(message, dict):
            raise InferenceError("Unexpected response format: message must be an object")

        return self._Result(
            message=dict(message),
            usage=self._parse_usage(body.get("usage")),
        )

    def _parse_usage(self, raw_usage: object) -> _Usage:
        if not isinstance(raw_usage, dict):
            return self._Usage(None, None)

        def token_or_none(name: str) -> int | None:
            value = raw_usage.get(name)
            return value if type(value) is int and value >= 0 else None

        return self._Usage(
            input_tokens=token_or_none("prompt_tokens"),
            output_tokens=token_or_none("completion_tokens"),
        )

    def _record_generation(
        self,
        messages: list[dict[str, Any]],
        result: _Result,
        span_name: str | None,
    ) -> None:
        get_client().update_current_generation(
            name=span_name,
            model=self.model,
            input=messages,
            output=result.message,
            usage_details=result.usage.as_langfuse(),
        )

    @observe(as_type="generation")
    def generate(
        self,
        messages: str | list[dict[str, Any]],
        *,
        reasoning: bool = True,
        last_response: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
        span_name: str | None = None,
    ) -> dict[str, Any]:
        """Generate a response and return the assistant message dict.

        Args:
            messages: A user prompt string or a list of message dicts.
            reasoning: Whether to enable model reasoning.
            last_response: Previous assistant response to prepend (preserves
                reasoning_details for multi-turn reasoning chains).
        """
        payload_messages = self._prepare_messages(messages, last_response)
        payload = self._build_payload(payload_messages, reasoning, schema)
        body = self._request_completion(payload)
        result = self._parse_response(body)
        self._record_generation(payload_messages, result, span_name)
        return result.message
