import json
import logging
import warnings

import pytest

with warnings.catch_warnings():
    # starlette warns at import that httpx2 is preferred; adding a dependency is out of scope
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from evalh.clients.base import ChatResult
from gateway.app import build_default_app, create_app
from gateway.breaker import CircuitBreaker
from gateway.config import Backend, ConfigError, GatewayConfig, Route, load_config, validate
from gateway.router import Router

TOOL = [{"type": "function", "function": {"name": "get_bgp_session", "parameters": {
    "type": "object", "properties": {"neighbor": {"type": "string"}}, "required": ["neighbor"]}}}]


class Scripted:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    async def chat(self, model, messages, tools=None, options=None, **kw):
        self.calls += 1
        return self.results.pop(0) if self.results else ChatResult(content="ok")


def cfg(chain, data_class="public"):
    backends = {"a": Backend("a", "ollama", "m-a"), "b": Backend("b", "ollama", "m-b"),
                "h": Backend("h", "openai_compat", "gpt")}
    return GatewayConfig(backends, {"r": Route("r", data_class, chain, 3, 20)})


async def test_primary_success():
    router = Router(cfg(["a", "b"]), {"a": Scripted(ChatResult(content="hi")), "b": Scripted()}, CircuitBreaker())
    out = await router.chat("r", [{"role": "user", "content": "x"}])
    assert out.served_by == "a" and out.attempts == []


@pytest.mark.parametrize("error,reason", [("timeout", "timeout"), ("first_token_timeout", "first_token_timeout"),
                                          ("http 500: boom", "error"), ("ollama: model not found", "error"),
                                          ("api: bad", "error"), ("JSONDecodeError: x", "error")])
async def test_fallback_reasons(error, reason):
    router = Router(cfg(["a", "b"]), {"a": Scripted(ChatResult.failed(error, 1.0)), "b": Scripted()}, CircuitBreaker())
    out = await router.chat("r", [])
    assert out.served_by == "b" and out.attempts == [("a", reason)]


async def test_invalid_tool_call_falls_back():
    bad = ChatResult(tool_calls=[{"name": "get_bgp_session", "arguments": {}}])
    router = Router(cfg(["a", "b"]), {"a": Scripted(bad), "b": Scripted()}, CircuitBreaker())
    out = await router.chat("r", [], tools=TOOL)
    assert out.served_by == "b" and out.attempts == [("a", "invalid_tool_call")]


async def test_all_fail():
    router = Router(cfg(["a"]), {"a": Scripted(ChatResult.failed("timeout", 1))}, CircuitBreaker())
    out = await router.chat("r", [])
    assert out.result is None and out.served_by is None


def test_breaker_opens_and_half_opens():
    now = [0.0]
    b = CircuitBreaker(threshold=2, cooldown_s=30, clock=lambda: now[0])
    b.record("a", False)
    assert b.allow("a")
    b.record("a", False)
    assert not b.allow("a")
    now[0] = 31
    assert b.allow("a")
    b.record("a", True)
    assert b.allow("a")


async def test_open_circuit_is_skipped():
    breaker = CircuitBreaker(threshold=1)
    breaker.record("a", False)
    a = Scripted()
    router = Router(cfg(["a", "b"]), {"a": a, "b": Scripted()}, breaker)
    out = await router.chat("r", [])
    assert a.calls == 0 and out.attempts == [("a", "circuit_open")]


def test_internal_route_cannot_use_hosted():
    with pytest.raises(ConfigError):
        validate(cfg(["a", "h"], data_class="internal"))


async def test_internal_route_blocked_at_runtime_too():
    router = Router(cfg(["h"], data_class="internal"), {"h": Scripted()}, CircuitBreaker(), check=False)
    out = await router.chat("r", [])
    assert out.result is None and out.attempts == [("h", "policy_blocked")]


async def test_missing_client_is_error():
    router = Router(cfg(["h", "a"]), {"a": Scripted()}, CircuitBreaker())
    out = await router.chat("r", [])
    assert out.served_by == "a" and out.attempts == [("h", "error")]


async def test_log_lines_are_json(caplog):
    caplog.set_level(logging.INFO, logger="gateway")
    router = Router(cfg(["a"]), {"a": Scripted(ChatResult.failed("timeout", 1))}, CircuitBreaker())
    await router.chat("r", [])
    router = Router(cfg(["a"]), {"a": Scripted()}, CircuitBreaker())
    await router.chat("r", [])
    msgs = [r.getMessage() for r in caplog.records if r.name == "gateway"]
    assert len(msgs) == 2
    for m in msgs:
        json.loads(m)


def test_http_api_headers():
    router = Router(cfg(["a", "b"]), {"a": Scripted(ChatResult.failed("timeout", 1)), "b": Scripted()}, CircuitBreaker())
    client = TestClient(create_app(router))
    r = client.post("/v1/chat", json={"route": "r", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200
    assert r.headers["x-served-by"] == "b" and r.headers["x-fallback-reason"] == "timeout"
    assert "gateway_fallbacks_total" in client.get("/metrics").text
    assert client.post("/v1/chat", json={"route": "nope", "messages": []}).status_code == 404


def test_multiple_apps_no_duplicate_metrics():
    for _ in range(2):
        create_app(Router(cfg(["a"]), {"a": Scripted()}, CircuitBreaker()))


def test_all_fail_returns_503():
    router = Router(cfg(["a"]), {"a": Scripted(ChatResult.failed("timeout", 1))}, CircuitBreaker())
    r = TestClient(create_app(router)).post("/v1/chat", json={"route": "r", "messages": []})
    assert r.status_code == 503 and r.headers["x-fallback-reason"] == "timeout"


def test_shipped_routes_load():
    assert set(load_config("configs/routes.yaml").routes) == {"noc-assistant", "doc-qa", "triage"}


def test_default_app_closes_clients_and_skips_hosted(monkeypatch):
    for k in ("HOSTED_BASE_URL", "HOSTED_API_KEY", "HOSTED_MODEL"):
        monkeypatch.delenv(k, raising=False)
    app = build_default_app()
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"ok": True}
        router = app.state.router
        assert "hosted" not in router.clients and "qwen7b" in router.clients
        http = router.clients["qwen7b"]._http
        assert not http.is_closed
    assert http.is_closed
