"""The model client: one agent loop over Ollama, OpenAI-compatible and Anthropic APIs."""
import json

import httpx
import pytest

import janus.llm as llm_mod
from janus.llm import LLM, LLMError, _extract_json, _split_system


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)


def _llm(settings, handler, **cfg):
    for k, v in cfg.items():
        setattr(settings, k, v)
    seen = []

    def record(request: httpx.Request):
        body = json.loads(request.content) if request.content else None
        seen.append((request, body))
        return handler(request, body)

    return LLM(settings, transport=httpx.MockTransport(record)), seen


MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]


def test_ollama_request_and_tokens(settings):
    llm, seen = _llm(settings, lambda r, b: httpx.Response(200, json={
        "message": {"content": '{"a": 1}'}, "prompt_eval_count": 5, "eval_count": 3}),
        provider="ollama", model="llama3", num_ctx=8192, seed=7)
    assert llm.chat_json(MSGS) == {"a": 1}
    req, body = seen[0]
    assert req.url.path == "/api/chat"
    assert body["format"] == "json"
    assert body["options"] == {"temperature": settings.temperature, "num_ctx": 8192, "seed": 7}
    assert llm.tokens_used == 8


def test_openai_compatible_request(settings):
    llm, seen = _llm(settings, lambda r, b: httpx.Response(200, json={
        "choices": [{"message": {"content": '{"ok": true}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2}}),
        provider="openai", model="gpt-x", api_base="https://example.test/v1/",
        api_key="sk-test", num_predict=100)
    assert llm.chat_json(MSGS) == {"ok": True}
    req, body = seen[0]
    assert str(req.url) == "https://example.test/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test"
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == 100
    assert body["messages"] == MSGS
    assert llm.tokens_used == 12


def test_rejected_params_are_dropped_and_remembered(settings):
    def handler(r, body):
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message":
                "Unsupported value: 'temperature' does not support 0.7"}})
        if "response_format" in body:
            return httpx.Response(400, text="response_format is not supported by this model")
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    llm, seen = _llm(settings, handler, provider="openai", model="m", api_key="k")
    assert llm.chat_json(MSGS, retries=0) == {}
    assert len(seen) == 3
    llm.chat_json(MSGS, retries=0)
    assert len(seen) == 4  # second call goes straight through
    assert "temperature" not in seen[-1][1] and "response_format" not in seen[-1][1]


def test_max_tokens_falls_back_to_max_completion_tokens(settings):
    def handler(r, body):
        if "max_tokens" in body:
            return httpx.Response(400, text="Unsupported parameter: 'max_tokens'. "
                                            "Use 'max_completion_tokens' instead.")
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    llm, seen = _llm(settings, handler, provider="openai", model="m", api_key="k",
                     num_predict=50)
    llm.chat_json(MSGS, retries=0)
    assert seen[-1][1]["max_completion_tokens"] == 50


def test_anthropic_request(settings):
    llm, seen = _llm(settings, lambda r, b: httpx.Response(200, json={
        "content": [{"type": "text", "text": 'Sure: {"x": 2}'}],
        "usage": {"input_tokens": 4, "output_tokens": 6}}),
        provider="anthropic", model="claude-x", api_key="ak")
    msgs = MSGS + [{"role": "user", "content": "again"}]  # consecutive user turns
    assert llm.chat_json(msgs) == {"x": 2}
    req, body = seen[0]
    assert str(req.url) == "https://api.anthropic.com/v1/messages"
    assert req.headers["x-api-key"] == "ak" and "anthropic-version" in req.headers
    assert body["system"].startswith("sys") and "JSON" in body["system"]
    assert body["messages"] == [{"role": "user", "content": "go\n\nagain"}]
    assert body["max_tokens"] > 0
    assert llm.tokens_used == 10


def test_auth_error_is_not_retried(settings):
    llm, seen = _llm(settings, lambda r, b: httpx.Response(401, text="bad key"),
                     provider="openai", model="m", api_key="k")
    with pytest.raises(LLMError, match="401"):
        llm.chat(MSGS, retries=3)
    assert len(seen) == 1


def test_rate_limit_is_retried(settings):
    responses = [httpx.Response(429, text="slow down"),
                 httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})]
    llm, seen = _llm(settings, lambda r, b: responses.pop(0),
                     provider="openai", model="m", api_key="k")
    assert llm.chat_json(MSGS) == {}
    assert len(seen) == 2


def test_unknown_provider(settings):
    llm, _ = _llm(settings, lambda r, b: httpx.Response(200), provider="nope")
    with pytest.raises(LLMError, match="unknown provider"):
        llm.chat(MSGS)
    assert not llm.ping()[0]


def test_ping_ollama_accepts_latest_tag(settings):
    llm, _ = _llm(settings, lambda r, b: httpx.Response(200, json={
        "models": [{"name": "llama3:latest"}]}), provider="ollama", model="llama3")
    assert llm.ping()[0]
    settings.model = "mistral"
    ok, msg = llm.ping()
    assert not ok and "ollama pull mistral" in msg


def test_ping_needs_api_key(settings, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm, seen = _llm(settings, lambda r, b: httpx.Response(200), provider="openai",
                     model="m", api_key="")
    ok, msg = llm.ping()
    assert not ok and "API key" in msg and not seen
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    llm2, seen2 = _llm(settings, lambda r, b: httpx.Response(200, json={"data": [{"id": "m"}]}),
                       provider="openai", model="m", api_key="")
    assert llm2.ping()[0]
    assert seen2[0][0].headers["authorization"] == "Bearer env-key"


def test_ping_server_without_model_list(settings):
    llm, _ = _llm(settings, lambda r, b: httpx.Response(404), provider="openai",
                  model="local", api_base="http://localhost:8080/v1", api_key="x")
    assert llm.ping()[0]


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('<think>maybe {"a": 0}? no</think>\n{"a": 1}', {"a": 1}),
    ('Here you go: {"a": {"b": 2}} hope that helps {"c": 3}', {"a": {"b": 2}}),
    ('[1, 2] then {"a": 1}', {"a": 1}),
])
def test_extract_json(text, expected):
    assert _extract_json(text) == expected


def test_extract_json_rejects_non_json():
    with pytest.raises(ValueError):
        _extract_json("no json here")


def test_split_system_starts_with_user():
    system, turns = _split_system([{"role": "system", "content": "s"},
                                   {"role": "assistant", "content": "a"}])
    assert system == "s"
    assert [t["role"] for t in turns] == ["user", "assistant"]


def test_lineage_only_overrides_ollama_model(settings, monkeypatch):
    from janus import cli
    from janus.train.registry import ModelRegistry
    monkeypatch.setattr(ModelRegistry, "current_tag", lambda self, default: "janus:v3")
    settings.provider, settings.model = "ollama", "base"
    assert cli._active_model(settings) == "janus:v3"
    settings.provider, settings.model = "openai", "gpt-x"
    assert cli._active_model(settings) == "gpt-x"
