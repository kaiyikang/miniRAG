import json
import os
import re
from typing import Any, Dict, Optional

import requests

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMError(Exception):
    pass


class JSONParseError(Exception):
    pass


def call_llm(
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    timeout: int = 60,
) -> str:
    """
    Call OpenRouter and return raw text content.

    This function does not parse JSON.
    It only returns the model's text output.
    """

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMError(
            "OPENROUTER_API_KEY is missing. "
            "Make sure it is exported in your shell, for example: "
            "export OPENROUTER_API_KEY='...'"
        )

    model = model or os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")

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
                "content": prompt,
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
        raise LLMError(f"OpenRouter request failed: {exc}") from exc

    if response.status_code >= 400:
        raise LLMError(f"OpenRouter API error {response.status_code}: {response.text}")

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected OpenRouter response format: {data}") from exc


def parse_json(raw: str) -> Dict[str, Any]:
    """
    Parse JSON from model output.

    It supports:
    1. pure JSON
    2. JSON wrapped in ```json ... ```
    3. text that contains one JSON object
    """

    if not raw or not raw.strip():
        raise JSONParseError("Empty LLM output, cannot parse JSON.")

    text = raw.strip()

    # Case 1: direct JSON
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise JSONParseError(f"Expected JSON object, got {type(parsed).__name__}")
        return parsed
    except json.JSONDecodeError:
        pass

    # Case 2: markdown code block
    code_block_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.DOTALL,
    )
    if code_block_match:
        candidate = code_block_match.group(1)
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                raise JSONParseError(
                    f"Expected JSON object, got {type(parsed).__name__}"
                )
            return parsed
        except json.JSONDecodeError as exc:
            raise JSONParseError(
                f"Failed to parse JSON code block: {candidate}"
            ) from exc

    # Case 3: extract first JSON object
    object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if object_match:
        candidate = object_match.group(0)
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                raise JSONParseError(
                    f"Expected JSON object, got {type(parsed).__name__}"
                )
            return parsed
        except json.JSONDecodeError as exc:
            raise JSONParseError(
                f"Failed to parse extracted JSON: {candidate}"
            ) from exc

    raise JSONParseError(f"No valid JSON object found in LLM output: {raw}")


def call_llm_json(
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1000,
) -> Dict[str, Any]:
    """
    Convenience wrapper:

        raw = call_llm(prompt)
        result = parse_json(raw)

    This is the version you can use inside agents.
    """

    raw = call_llm(
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    print(f">>> DEBUG: {raw}")
    return parse_json(raw)
