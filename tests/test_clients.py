import json

import httpx
import pytest
import respx

from evalh.clients.ollama import OllamaClient
from evalh.clients.openai_compat import OpenAICompatClient


def ndjson(*chunks):
    return "".join(json.dumps(c) + "\n" for c in chunks).encode()


@respx.mock
async def test_ollama_stream_collects_text_and_stats():
    respx.post("http://ollama/api/chat").mock(return_value=httpx.Response(200, content=ndjson(
        {"message": {"role": "assistant", "content": "Hel"}, "done": False},
        {"message": {"role": "assistant", "content": "lo"}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True,
         "prompt_eval_count": 100, "prompt_eval_duration": 500_000_000, "prompt_eval_cached_count": 40,
         "eval_count": 20, "eval_duration": 1_000_000_000, "load_duration": 2_000_000_000},
    )))
    r = await OllamaClient("http://ollama").chat("m", [{"role": "user", "content": "hi"}])
    assert r.ok and r.content == "Hello"
    assert r.prefill_tps == pytest.approx(200.0) and r.decode_tps == pytest.approx(20.0)
    assert r.cached_prompt_tokens == 40 and r.load_s == pytest.approx(2.0)
    assert r.ttft_s is not None and r.ttft_s <= r.total_s


@respx.mock
async def test_ollama_tool_calls():
    respx.post("http://ollama/api/chat").mock(return_value=httpx.Response(200, content=ndjson(
        {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_bgp_session", "arguments": {"neighbor": "192.0.2.1"}}}]}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True, "eval_count": 5, "eval_duration": 10**8},
    )))
    r = await OllamaClient("http://ollama").chat("m", [], tools=[{"type": "function"}])
    assert r.tool_calls == [{"name": "get_bgp_session", "arguments": {"neighbor": "192.0.2.1"}}]


@respx.mock
async def test_ollama_http_error_is_captured():
    respx.post("http://ollama/api/chat").mock(return_value=httpx.Response(404, json={"error": "model not found"}))
    r = await OllamaClient("http://ollama").chat("m", [])
    assert not r.ok and r.error.startswith("http 404")


@respx.mock
async def test_ollama_timeout_is_captured():
    respx.post("http://ollama/api/chat").mock(side_effect=httpx.ReadTimeout("slow"))
    r = await OllamaClient("http://ollama").chat("m", [])
    assert r.error == "timeout"


@respx.mock
async def test_openai_compat_assembles_streamed_tool_args():
    sse = "".join(f"data: {json.dumps(c)}\n\n" for c in [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "lookup_prefix_origin", "arguments": "{\"pre"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "fix\": \"203.0.113.0/24\"}"}}]}}]},
        {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 12}},
    ]) + "data: [DONE]\n\n"
    respx.post("http://api/v1/chat/completions").mock(return_value=httpx.Response(200, content=sse.encode()))
    r = await OpenAICompatClient("http://api/v1", "k", "gpt-x").chat("hosted", [])
    assert r.tool_calls == [{"name": "lookup_prefix_origin", "arguments": {"prefix": "203.0.113.0/24"}}]
    assert r.prompt_tokens == 50 and r.completion_tokens == 12


@respx.mock
async def test_openai_compat_keeps_malformed_args_as_string():
    sse = (f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'name': 'x', 'arguments': '{bad'}}]}}]})}\n\n"
           "data: [DONE]\n\n")
    respx.post("http://api/v1/chat/completions").mock(return_value=httpx.Response(200, content=sse.encode()))
    r = await OpenAICompatClient("http://api/v1", "k", "gpt-x").chat("hosted", [])
    assert r.tool_calls == [{"name": "x", "arguments": "{bad"}]
