import json
import os
import re
from typing import Any, Dict, Optional
import logging
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)
import time

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMCallError(Exception):
    """Base error for LLM calls."""


class LLMTransientError(LLMCallError):
    """Network error, timeout, rate limit, temporary provider issue."""


class LLMPermanentError(LLMCallError):
    """Bad request, unauthorized, forbidden, model not found."""


class LLMInvalidJSONError(LLMCallError):
    """Model returned text that cannot be parsed as JSON."""


class LLMProviderResponseError(LLMCallError):
    """Provider returned an unexpected API response shape."""


@retry(
    retry=retry_if_exception_type(LLMTransientError),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=8),
    reraise=True,
)
def call_llm(
    user_prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    timeout: int = 60,
) -> str:

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMCallError(
            "OPENROUTER_API_KEY is missing. "
            "Make sure it is exported in your shell, for example: "
            "export OPENROUTER_API_KEY='...'"
        )

    model = model or os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")

    start = time.perf_counter()
    logger.info("llm_call_started", extra={"model": model})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional, but useful for OpenRouter dashboard / rankings
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "multi-agent-rag-learning"),
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict JSON generator. "
                    "Return only valid JSON. No markdown. No explanation."
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise LLMTransientError(f"OpenRouter request failed: {exc}") from exc

    if response.status_code in {408, 429} or response.status_code >= 500:
        raise LLMTransientError(
            f"OpenRouter temporary error {response.status_code}: {response.text}"
        )

    if response.status_code >= 400:
        raise LLMPermanentError(
            f"OpenRouter permanent error {response.status_code}: {response.text}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise LLMProviderResponseError(
            f"OpenRouter returned non-JSON response: {response.text}"
        ) from exc

    try:
        choice = data["choices"][0]
        finish_reason = choice.get("finish_reason")
        content = choice["message"]["content"]

        if finish_reason == "length":
            raise LLMTransientError("LLM output was truncated. Increase max_tokens.")

        logger.info(
            "llm_call_succeeded",
            extra={
                "model": model,
                "elapsed_ms": int((time.perf_counter() - start) * 1000),
            },
        )
        return content
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise LLMProviderResponseError(
            f"Unexpected OpenRouter response format: {data}"
        ) from exc


def parse_json(raw: str, *, allow_extract: bool = False) -> Dict[str, Any]:

    if not raw or not raw.strip():
        raise LLMInvalidJSONError("Empty LLM output, cannot parse JSON.")

    text = raw.strip()

    # Case 1: direct JSON
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise LLMInvalidJSONError(
                f"Expected JSON object, got {type(parsed).__name__}"
            )
        return parsed
    except json.JSONDecodeError:
        pass

    # Case 2: markdown code block
    code_block_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if code_block_match:
        candidate = code_block_match.group(1)
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                raise LLMInvalidJSONError(
                    f"Expected JSON object, got {type(parsed).__name__}"
                )
            return parsed
        except json.JSONDecodeError as exc:
            raise LLMInvalidJSONError(
                f"Failed to parse JSON code block: {candidate}"
            ) from exc

    # Case 3: extract first JSON object
    if allow_extract:
        object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if object_match:
            candidate = object_match.group(0)
            try:
                parsed = json.loads(candidate)
                if not isinstance(parsed, dict):
                    raise LLMInvalidJSONError(
                        f"Expected JSON object, got {type(parsed).__name__}"
                    )
                return parsed
            except json.JSONDecodeError as exc:
                raise LLMInvalidJSONError(
                    f"Failed to parse extracted JSON: {candidate}"
                ) from exc

    raise LLMInvalidJSONError(f"No valid JSON object found in LLM output: {raw}")


def call_llm_json(
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    timeout: int = 60,
) -> Dict[str, Any]:

    raw = call_llm(
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    return parse_json(raw)
