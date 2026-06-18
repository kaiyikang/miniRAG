from abc import ABC, abstractmethod
from typing import Any

import requests

from minirag.config import get_settings


class InferenceEngine(ABC):
    @abstractmethod
    def generate(
        self,
        messages: str | list[dict[str, Any]],
        *,
        reasoning: bool = True,
        last_response: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a response and return the assistant message dict."""


class InferenceError(Exception):
    """Raised when the LLM inference request fails."""


class OpenRouterEngine(InferenceEngine):
    """LLM inference engine backed by the OpenRouter API, with reasoning support."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ):
        settings = get_settings()
        self.model = model or settings.openrouter_model
        self.api_key = api_key or settings.openrouter_api_key
        if not self.api_key:
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

    def generate(
        self,
        messages: str | list[dict[str, Any]],
        *,
        reasoning: bool = True,
        last_response: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a response and return the assistant message dict.

        Args:
            messages: A user prompt string or a list of message dicts.
            reasoning: Whether to enable model reasoning.
            last_response: Previous assistant response to prepend (preserves
                reasoning_details for multi-turn reasoning chains).
        """
        payload_messages = self._prepare_messages(messages, last_response)

        try:
            response = requests.post(
                self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": payload_messages,
                    "reasoning": {"enabled": reasoning},
                },
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise InferenceError(f"LLM inference failed: {exc}") from exc

        try:
            return response.json()["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError(f"Unexpected response format: {exc}") from exc
