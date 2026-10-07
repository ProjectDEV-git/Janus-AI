"""Single choke point for all model calls, via Ollama.

Keeping every model interaction here means swapping models, adding retries, or
accounting tokens happens in exactly one place.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import httpx

from janus.config import Settings


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLM:
    """Thin Ollama chat client with token accounting and retry."""

    settings: Settings
    tokens_used: int = field(default=0, init=False)

    def _options(self) -> dict:
        opts: dict = {"temperature": self.settings.temperature}
        if self.settings.num_predict > 0:
            opts["num_predict"] = self.settings.num_predict
        return opts

    def chat(
        self,
        messages: list[dict],
        *,
        format_json: bool = False,
        retries: int = 2,
    ) -> ChatResult:
        """Call /api/chat (non-streaming). Raises RuntimeError on repeated failure."""
        payload: dict = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
            "options": self._options(),
        }
        if format_json:
            payload["format"] = "json"

        url = self.settings.ollama_host.rstrip("/") + "/api/chat"
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            start = time.monotonic()
            try:
                with httpx.Client(timeout=300.0) as client:
                    resp = client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                content = (data.get("message") or {}).get("content", "")
                res = ChatResult(
                    content=content,
                    prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
                    completion_tokens=int(data.get("eval_count", 0) or 0),
                    duration_s=time.monotonic() - start,
                )
                self.tokens_used += res.total_tokens
                return res
            except Exception as exc:  # noqa: BLE001 - surfaced after retries
                last_err = exc
                if attempt < retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Ollama chat failed after {retries + 1} attempts: {last_err}")

    def chat_json(self, messages: list[dict], *, retries: int = 2) -> dict:
        """Chat expecting a JSON object back; tolerant of fenced/surrounding text."""
        res = self.chat(messages, format_json=True, retries=retries)
        return _extract_json(res.content)

    def ping(self) -> tuple[bool, str]:
        """Check Ollama reachability and whether the configured model is present."""
        base = self.settings.ollama_host.rstrip("/")
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(base + "/api/tags")
                resp.raise_for_status()
                tags = resp.json().get("models", [])
        except Exception as exc:  # noqa: BLE001
            return False, f"cannot reach Ollama at {base}: {exc}"
        names = {m.get("name", "") for m in tags}
        if self.settings.model in names:
            return True, f"ok: model '{self.settings.model}' available"
        return (
            False,
            f"Ollama reachable but model '{self.settings.model}' not pulled. "
            f"Run: ollama pull {self.settings.model}",
        )


def _extract_json(text: str) -> dict:
    """Best-effort parse of a JSON object out of model text."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"model did not return valid JSON: {text[:200]!r}")
