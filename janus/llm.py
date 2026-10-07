"""Single choke point for all model calls.

Keeping every model interaction here means swapping models, adding retries, or
accounting tokens happens in exactly one place. Three wire protocols cover
practically every model:

  - "ollama"    — a local Ollama server (/api/chat).
  - "openai"    — any OpenAI-compatible /chat/completions endpoint: OpenAI,
                  OpenRouter, Groq, Together, Mistral, DeepSeek, Gemini's OpenAI
                  endpoint, LM Studio, vLLM, llama.cpp's server, LiteLLM, ...
  - "anthropic" — Anthropic's Messages API (/v1/messages).

Models differ in which knobs they accept (JSON mode, temperature, seed, ...).
Optional parameters a server rejects are dropped and remembered, so the same
agent loop runs unchanged on any of them.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import httpx

from janus.config import Settings

PROVIDERS = ("ollama", "openai", "anthropic")

_DEFAULT_BASE = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
}
_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
_ANTHROPIC_VERSION = "2023-06-01"
_ANTHROPIC_DEFAULT_MAX_TOKENS = 4096
_JSON_HINT = "Respond with a single JSON object only."

# Optional request parameters that some models reject. Required ones (model,
# messages, Anthropic's max_tokens) are never dropped.
_DROPPABLE = {
    "openai": ("response_format", "temperature", "seed", "max_tokens", "max_completion_tokens"),
    "anthropic": ("temperature",),
    "ollama": (),
}


class LLMError(RuntimeError):
    """A model call failed in a way retrying will not fix (auth, unknown model, ...)."""


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
    """Provider-agnostic chat client with token accounting and retry."""

    settings: Settings
    transport: httpx.BaseTransport | None = None  # injectable for tests
    tokens_used: int = field(default=0, init=False)
    _rejected: set = field(default_factory=set, init=False)  # params this server refused

    # --- public API -------------------------------------------------------

    @property
    def provider(self) -> str:
        p = (self.settings.provider or "ollama").lower()
        if p not in PROVIDERS:
            raise LLMError(f"unknown provider {p!r}; use one of {', '.join(PROVIDERS)}")
        return p

    def chat(self, messages: list[dict], *, format_json: bool = False,
             retries: int = 2) -> ChatResult:
        """Send a chat. Retries network errors, 429 and 5xx; raises on failure."""
        provider = self.provider
        last_err: Exception | None = None
        attempt = 0
        while attempt <= retries:
            url, headers, payload = self._request(provider, messages, format_json)
            start = time.monotonic()
            try:
                with self._client(timeout=300.0) as client:
                    resp = client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last_err = exc
            else:
                if resp.status_code == 200:
                    res = self._parse(provider, resp.json())
                    res.duration_s = time.monotonic() - start
                    self.tokens_used += res.total_tokens
                    return res
                if resp.status_code == 400 and self._drop_rejected(provider, payload, resp.text):
                    continue  # retry without the parameter; doesn't use up an attempt
                last_err = LLMError(f"HTTP {resp.status_code} from {url}: {resp.text[:300]}")
                if resp.status_code not in (408, 409, 429) and resp.status_code < 500:
                    raise last_err  # auth, bad model name, ...: retrying won't help
            attempt += 1
            if attempt <= retries:
                time.sleep(2 ** (attempt - 1))
        raise LLMError(f"{provider} chat failed after {retries + 1} attempts: {last_err}")

    def chat_json(self, messages: list[dict], *, retries: int = 2) -> dict:
        """Chat expecting a JSON object back; tolerant of fences, prose and <think> blocks."""
        res = self.chat(messages, format_json=True, retries=retries)
        return _extract_json(res.content)

    def ping(self) -> tuple[bool, str]:
        """Check the endpoint is reachable and the configured model exists."""
        try:
            provider = self.provider
        except LLMError as e:
            return False, str(e)
        model = self.settings.model
        if provider != "ollama" and not self._api_key():
            return False, (f"no API key for {provider}: set JANUS_API_KEY or "
                           f"{_KEY_ENV[provider]}")
        url, headers = self._models_endpoint(provider)
        try:
            with self._client(timeout=15.0) as client:
                resp = client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            return False, f"cannot reach {provider} at {url}: {exc}"
        if resp.status_code in (401, 403):
            return False, f"{provider} rejected the API key (HTTP {resp.status_code})"
        if resp.status_code != 200:
            # Many OpenAI-compatible servers don't implement /models; reachable is enough.
            return True, f"ok: {provider} reachable (model list unavailable; using '{model}')"
        names = _model_names(provider, resp.json())
        if _model_listed(model, names):
            return True, f"ok: {provider} model '{model}' available"
        if provider == "ollama":
            return False, (f"Ollama reachable but model '{model}' not pulled. "
                           f"Run: ollama pull {model}")
        if not names:
            return True, f"ok: {provider} reachable (using '{model}')"
        return False, (f"{provider} reachable but has no model '{model}'. "
                       f"Some available: {', '.join(sorted(names)[:8])}")

    # --- providers --------------------------------------------------------

    def _client(self, timeout: float) -> httpx.Client:
        return httpx.Client(timeout=timeout, transport=self.transport)

    def _api_key(self) -> str:
        if self.settings.api_key:
            return self.settings.api_key
        return os.environ.get(_KEY_ENV.get(self.provider, ""), "")

    def _base(self, provider: str) -> str:
        if provider == "ollama":
            return self.settings.ollama_host.rstrip("/")
        return (self.settings.api_base or _DEFAULT_BASE[provider]).rstrip("/")

    def _headers(self, provider: str) -> dict:
        key = self._api_key()
        if provider == "anthropic":
            return {"x-api-key": key, "anthropic-version": _ANTHROPIC_VERSION}
        if provider == "openai" and key:
            return {"Authorization": f"Bearer {key}"}
        return {}

    def _models_endpoint(self, provider: str) -> tuple[str, dict]:
        base = self._base(provider)
        if provider == "ollama":
            return base + "/api/tags", {}
        if provider == "anthropic":
            return base + "/v1/models?limit=1000", self._headers(provider)
        return base + "/models", self._headers(provider)

    def _request(self, provider: str, messages: list[dict],
                 format_json: bool) -> tuple[str, dict, dict]:
        s = self.settings
        base, headers = self._base(provider), self._headers(provider)
        if provider == "ollama":
            opts: dict = {"temperature": s.temperature}
            if s.num_predict > 0:
                opts["num_predict"] = s.num_predict
            if s.num_ctx > 0:
                opts["num_ctx"] = s.num_ctx
            if s.seed is not None:
                opts["seed"] = s.seed
            payload = {"model": s.model, "messages": messages, "stream": False, "options": opts}
            if format_json:
                payload["format"] = "json"
            return base + "/api/chat", headers, payload

        if provider == "anthropic":
            system, turns = _split_system(messages)
            if format_json:
                system = f"{system}\n\n{_JSON_HINT}".strip()
            payload = {"model": s.model, "messages": turns,
                       "max_tokens": s.num_predict if s.num_predict > 0
                       else _ANTHROPIC_DEFAULT_MAX_TOKENS,
                       "temperature": s.temperature}
            if system:
                payload["system"] = system
            return base + "/v1/messages", headers, self._without_rejected(payload)

        payload = {"model": s.model, "messages": messages, "temperature": s.temperature}
        if s.num_predict > 0:
            payload["max_completion_tokens" if "max_tokens" in self._rejected
                    else "max_tokens"] = s.num_predict
        if s.seed is not None:
            payload["seed"] = s.seed
        if format_json:
            payload["response_format"] = {"type": "json_object"}
        return base + "/chat/completions", headers, self._without_rejected(payload)

    def _without_rejected(self, payload: dict) -> dict:
        return {k: v for k, v in payload.items() if k not in self._rejected}

    def _drop_rejected(self, provider: str, payload: dict, error_text: str) -> bool:
        """If a 400 names an optional parameter we sent, stop sending it. Returns True
        when something was dropped (so the call is worth retrying)."""
        text = error_text.lower()
        for param in _DROPPABLE[provider]:
            if param in payload and param not in self._rejected and param in text:
                self._rejected.add(param)
                return True
        return False

    @staticmethod
    def _parse(provider: str, data: dict) -> ChatResult:
        if provider == "ollama":
            return ChatResult(
                content=(data.get("message") or {}).get("content", ""),
                prompt_tokens=int(data.get("prompt_eval_count") or 0),
                completion_tokens=int(data.get("eval_count") or 0),
            )
        if provider == "anthropic":
            usage = data.get("usage") or {}
            text = "".join(b.get("text", "") for b in data.get("content") or []
                           if b.get("type") == "text")
            return ChatResult(text, int(usage.get("input_tokens") or 0),
                              int(usage.get("output_tokens") or 0))
        usage = data.get("usage") or {}
        choices = data.get("choices") or [{}]
        content = (choices[0].get("message") or {}).get("content") or ""
        return ChatResult(content, int(usage.get("prompt_tokens") or 0),
                          int(usage.get("completion_tokens") or 0))


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Anthropic takes the system prompt separately and needs strictly alternating
    user/assistant turns starting with user; merge consecutive same-role turns."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns: list[dict] = []
    for m in messages:
        if m["role"] == "system":
            continue
        if turns and turns[-1]["role"] == m["role"]:
            turns[-1] = {"role": m["role"], "content": turns[-1]["content"] + "\n\n" + m["content"]}
        else:
            turns.append({"role": m["role"], "content": m["content"]})
    if turns and turns[0]["role"] != "user":
        turns.insert(0, {"role": "user", "content": "Begin."})
    return system, turns


def _model_names(provider: str, data: dict) -> set[str]:
    if provider == "ollama":
        return {m.get("name", "") for m in data.get("models", [])} - {""}
    return {m.get("id", "") for m in data.get("data", [])} - {""}


def _model_listed(model: str, names: set[str]) -> bool:
    # Ollama lists "llama3:latest" for a model pulled as "llama3".
    return model in names or (":" not in model and f"{model}:latest" in names)


_THINK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I)


def _extract_json(text: str) -> dict:
    """Best-effort parse of a JSON object out of model text.

    Handles markdown fences, prose around the object, and the <think> blocks that
    reasoning models (DeepSeek-R1, Qwen3, ...) emit before answering."""
    text = _THINK_RE.sub("", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"model did not return valid JSON: {text[:200]!r}")
