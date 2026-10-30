"""
OpenAI Chat Completions compatibility helpers.

This module keeps existing behavior for legacy models/providers while
gracefully adapting request parameters for GPT-5 family models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def is_gpt5_family(model: Optional[str]) -> bool:
    """Return True when model belongs to GPT-5 family aliases/snapshots."""
    if not model:
        return False
    return model.strip().lower().startswith("gpt-5")


def _create_chat_completion_once(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
) -> Any:
    """
    Create a chat completion with model-specific request parameters.

    Compatibility strategy:
    - For GPT-5 family, avoid sending temperature by default.
    - For token limit, use `max_completion_tokens` on GPT-5, `max_tokens` otherwise.
    - Preserve the legacy request shape for every non-GPT-5 model/provider.
    - Propagate provider errors unchanged instead of guessing from message text.
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    if response_format is not None:
        kwargs["response_format"] = response_format

    if tools:
        kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

    gpt5_family = is_gpt5_family(model)

    if temperature is not None and not gpt5_family:
        kwargs["temperature"] = temperature

    if max_tokens is not None:
        if gpt5_family:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

    return client.chat.completions.create(**kwargs)


def extract_chat_completion_text(response: Any) -> str:
    """Extract plain text from chat completion response across SDK content shapes."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text_obj = item.get("text")
                if isinstance(text_obj, dict):
                    text_obj = text_obj.get("value")
                if isinstance(text_obj, str):
                    chunks.append(text_obj)
                elif isinstance(item.get("content"), str):
                    chunks.append(item["content"])
                continue

            text_obj = getattr(item, "text", None)
            if isinstance(text_obj, dict):
                text_obj = text_obj.get("value")
            if isinstance(text_obj, str):
                chunks.append(text_obj)
                continue

            content_obj = getattr(item, "content", None)
            if isinstance(content_obj, str):
                chunks.append(content_obj)

        return "".join(chunks).strip()

    return str(content or "")


# --- Token-per-minute budget -------------------------------------------------
# Groq's on-demand tier rejects any single request whose input plus max_tokens
# exceeds the model's TPM limit (HTTP 413), and throttles bursts with HTTP 429.
# Report generation accumulates tool observations, so requests are fitted to
# the budget before sending and retried with backoff when the provider objects.

import os as _os
import re as _re
import time as _time
import logging as _logging

_log = _logging.getLogger("mirofish.llm_budget")
_CHARS_PER_TOKEN = 3.0
_MIN_COMPLETION_TOKENS = 1024
_TRIM_MARKER = "\n…[truncated to fit the model token budget]…\n"


def _tpm_limit(client: Any) -> int:
    raw = _os.environ.get("MIROFISH_LLM_TPM_LIMIT", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            return 0
    return 8000 if "groq.com" in str(getattr(client, "base_url", "")) else 0


def _content_len(message: Dict[str, Any]) -> int:
    content = message.get("content")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(str(part.get("text", ""))) for part in content if isinstance(part, dict))
    return 0


def _estimate_tokens(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> int:
    chars = sum(_content_len(m) + 16 for m in messages)
    for m in messages:
        for call in m.get("tool_calls") or []:
            chars += len(str(call))
    if tools:
        chars += len(str(tools))
    return int(chars / _CHARS_PER_TOKEN) + 64


def _trim_messages(messages: List[Dict[str, Any]], excess_tokens: int) -> List[Dict[str, Any]]:
    """Shorten the longest string contents (never the first system prompt or
    the final message) by keeping their head and tail."""
    trimmed = [dict(m) for m in messages]
    excess_chars = int(excess_tokens * _CHARS_PER_TOKEN) + 200
    protected = {0, len(trimmed) - 1} if len(trimmed) > 2 else set()
    while excess_chars > 0:
        candidates = [
            (len(m["content"]), i) for i, m in enumerate(trimmed)
            if i not in protected and isinstance(m.get("content"), str) and len(m["content"]) > 1200
        ]
        if not candidates:
            break
        length, i = max(candidates)
        cut = min(excess_chars + len(_TRIM_MARKER), length - 800)
        if cut <= len(_TRIM_MARKER):
            break
        keep = length - cut
        head = int(keep * 0.7)
        content = trimmed[i]["content"]
        trimmed[i]["content"] = content[:head] + _TRIM_MARKER + content[length - (keep - head):]
        excess_chars -= cut - len(_TRIM_MARKER)
    return trimmed


def _fit_to_budget(limit, messages, tools, max_tokens):
    target = int(limit * 0.88)
    est = _estimate_tokens(messages, tools)
    completion = max_tokens or 4096
    if est + completion <= target:
        return messages, max_tokens
    completion = max(_MIN_COMPLETION_TOKENS, min(completion, target - est))
    if est + completion > target:
        messages = _trim_messages(messages, est + completion - target)
    return messages, completion


def _rate_limit_kind(error: Exception) -> Optional[str]:
    status = getattr(error, "status_code", None)
    text = str(error)
    if "rate_limit_exceeded" not in text and status not in (413, 429):
        return None
    if status == 413 or "Request too large" in text:
        return "too_large"
    return "throttled"


def _retry_after_seconds(error: Exception, default: float) -> float:
    match = _re.search(r"try again in ([0-9.]+)(ms|s)", str(error))
    if match:
        value = float(match.group(1))
        return min(60.0, value / 1000 if match.group(2) == "ms" else value) + 0.5
    response = getattr(error, "response", None)
    header = getattr(response, "headers", {}).get("retry-after") if response is not None else None
    try:
        return min(60.0, float(header)) + 0.5 if header else default
    except (TypeError, ValueError):
        return default


def create_chat_completion(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
) -> Any:
    limit = _tpm_limit(client)
    if limit:
        messages, max_tokens = _fit_to_budget(limit, messages, tools, max_tokens)

    attempts = 4 if limit else 1
    for attempt in range(attempts):
        try:
            return _create_chat_completion_once(
                client,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
            )
        except Exception as error:
            kind = _rate_limit_kind(error) if limit else None
            if kind is None or attempt == attempts - 1:
                raise
            if kind == "too_large":
                limit = int(limit * 0.8)
                messages, max_tokens = _fit_to_budget(limit, messages, tools, max_tokens)
                delay = 1.0
            else:
                delay = _retry_after_seconds(error, default=8.0 * (attempt + 1))
            _log.warning("LLM rate limit (%s), retrying in %.1fs (attempt %d)", kind, delay, attempt + 2)
            _time.sleep(delay)
