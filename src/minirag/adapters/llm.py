from typing import Any
from minirag.domain.ports import InferenceEngine, InferenceError
import requests
from langfuse import get_client, observe


class OpenRouterEngine(InferenceEngine):
    """LLM inference engine backed by the OpenRouter API, with reasoning support."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str,
        api_key: str,
    ):
        self.model = model
        self.api_key = api_key
        if not self.api_key or not self.model:
            raise RuntimeError(
                "OpenRouter API key is required. Pass api_key=... or set OPENROUTER_API_KEY."
            )

    def _prepare_messages(
        self, messages: str | list[dict[str, Any]], last_response: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        if last_response is not None:
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": last_response.get("content"),
            }
            reasoning = last_response.get("reasoning_details")
            if reasoning is not None:
                assistant_msg["reasoning_details"] = reasoning
            messages = [assistant_msg, *messages]

        return messages

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

        response_format = (
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "miniRAG",
                    "strict": True,
                    "schema": schema,
                },
            }
            if schema
            else None
        )

        payload = {
            "model": self.model,
            "messages": payload_messages,
            "reasoning": {"enabled": reasoning},
        }

        if response_format:
            payload["response_format"] = response_format

        try:
            response = requests.post(
                self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise InferenceError(f"LLM inference failed: {exc}") from exc

        try:
            body = response.json()
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError(f"Unexpected response format: {exc}") from exc

        # Normalise OpenRouter's usage keys (prompt_tokens/completion_tokens) to
        # Langfuse's standard input/output so they match the model definition's
        # price keys
        raw_usage = body.get("usage") or {}
        usage_details = {
            k: v
            for k, v in {
                "input": raw_usage.get("prompt_tokens"),
                "output": raw_usage.get("completion_tokens"),
            }.items()
            if v is not None
        }

        get_client().update_current_generation(
            name=span_name,
            model=self.model,
            input=payload_messages,
            output=message,
            usage_details=usage_details or None,
        )
        return message
