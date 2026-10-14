import asyncio
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
    async with OllamaClient("http://ollama") as c:
        r = await c.chat("m", [{"role": "user", "content": "hi"}])
    assert r.ok and r.content == "Hello"
    # 100 prompt tokens, 40 served from cache: only 60 were actually prefilled in 0.5 s
    assert r.prefill_tps == pytest.approx(120.0) and r.decode_tps == pytest.approx(20.0)
    assert r.cached_prompt_tokens == 40 and r.load_s == pytest.approx(2.0)
    assert r.ttft_s is not None and r.ttft_s <= r.total_s


@respx.mock
async def test_ollama_tool_calls():
    respx.post("http://ollama/api/chat").mock(return_value=httpx.Response(200, content=ndjson(
        {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_bgp_session", "arguments": {"neighbor": "192.0.2.1"}}}]}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True, "eval_count": 5, "eval_duration": 10**8},
    )))
    async with OllamaClient("http://ollama") as c:
        r = await c.chat("m", [], tools=[{"type": "function"}])
    assert r.tool_calls == [{"name": "get_bgp_session", "arguments": {"neighbor": "192.0.2.1"}}]


@respx.mock
async def test_ollama_http_error_is_captured():
    respx.post("http://ollama/api/chat").mock(return_value=httpx.Response(404, json={"error": "model not found"}))
    async with OllamaClient("http://ollama") as c:
        r = await c.chat("m", [])
    assert not r.ok and r.error.startswith("http 404")


@respx.mock
async def test_ollama_timeout_is_captured():
    respx.post("http://ollama/api/chat").mock(side_effect=httpx.ReadTimeout("slow"))
    async with OllamaClient("http://ollama") as c:
        r = await c.chat("m", [])
    assert r.error == "timeout"


@respx.mock
async def test_openai_compat_assembles_streamed_tool_args():
    sse = "".join(f"data: {json.dumps(c)}\n\n" for c in [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "lookup_prefix_origin", "arguments": "{\"pre"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "fix\": \"203.0.113.0/24\"}"}}]}}]},
        {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 12}},
    ]) + "data: [DONE]\n\n"
    respx.post("http://api/v1/chat/completions").mock(return_value=httpx.Response(200, content=sse.encode()))
    async with OpenAICompatClient("http://api/v1", "k", "gpt-x") as c:
        r = await c.chat("hosted", [])
    assert r.tool_calls == [{"name": "lookup_prefix_origin", "arguments": {"prefix": "203.0.113.0/24"}}]
    assert r.prompt_tokens == 50 and r.completion_tokens == 12


@respx.mock
async def test_openai_compat_keeps_malformed_args_as_string():
    sse = (f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'name': 'x', 'arguments': '{bad'}}]}}]})}\n\n"
           "data: [DONE]\n\n")
    respx.post("http://api/v1/chat/completions").mock(return_value=httpx.Response(200, content=sse.encode()))
    async with OpenAICompatClient("http://api/v1", "k", "gpt-x") as c:
        r = await c.chat("hosted", [])
    assert r.tool_calls == [{"name": "x", "arguments": "{bad"}]


@respx.mock
async def test_ollama_prefill_tps_excludes_cached_tokens():
    respx.post("http://ollama/api/chat").mock(return_value=httpx.Response(200, content=ndjson(
        {"message": {"content": "x"}, "done": False},
        {"message": {"content": ""}, "done": True, "prompt_eval_count": 306,
         "prompt_eval_cached_count": 305, "prompt_eval_duration": 10_000_000},
    )))
    async with OllamaClient("http://ollama") as c:
        r = await c.chat("m", [])
    assert r.prompt_tokens == 306 and r.cached_prompt_tokens == 305
    assert r.prefill_tps == pytest.approx(100.0)  # 1 new token in 10 ms, not 306 in 10 ms


@respx.mock
async def test_ollama_prefill_tps_is_none_when_everything_cached():
    respx.post("http://ollama/api/chat").mock(return_value=httpx.Response(200, content=ndjson(
        {"message": {"content": "x"}, "done": False},
        {"message": {"content": ""}, "done": True, "prompt_eval_count": 50,
         "prompt_eval_cached_count": 50, "prompt_eval_duration": 10_000_000},
    )))
    async with OllamaClient("http://ollama") as c:
        r = await c.chat("m", [])
    assert r.prefill_tps is None


@respx.mock
async def test_ollama_malformed_ndjson_line_returns_result():
    respx.post("http://ollama/api/chat").mock(return_value=httpx.Response(
        200, content=b'{"message": {"content": "a"}, "done": false}\n{not json\n'))
    async with OllamaClient("http://ollama") as c:
        r = await c.chat("m", [])
    assert not r.ok and r.error.startswith("JSONDecodeError")


@respx.mock
async def test_ollama_error_chunk_is_failed_result():
    respx.post("http://ollama/api/chat").mock(return_value=httpx.Response(
        200, content=ndjson({"error": "llama runner process has terminated"})))
    async with OllamaClient("http://ollama") as c:
        r = await c.chat("m", [])
    assert r.error == "ollama: llama runner process has terminated"


@respx.mock
async def test_openai_compat_error_line_is_failed_result():
    sse = f"data: {json.dumps({'error': {'message': 'rate limited', 'type': 'x'}})}\n\n"
    respx.post("http://api/v1/chat/completions").mock(return_value=httpx.Response(200, content=sse.encode()))
    async with OpenAICompatClient("http://api/v1", "k", "gpt-x") as c:
        r = await c.chat("hosted", [])
    assert r.error == "api: rate limited"


@respx.mock
async def test_openai_compat_malformed_json_returns_result():
    respx.post("http://api/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=b"data: {oops\n\n"))
    async with OpenAICompatClient("http://api/v1", "k", "gpt-x") as c:
        r = await c.chat("hosted", [])
    assert not r.ok and r.error.startswith("JSONDecodeError")


async def test_aclose_closes_underlying_client():
    c = OllamaClient("http://ollama")
    await c.aclose()
    assert c._http.is_closed
    o = OpenAICompatClient("http://api/v1", "k", "m")
    await o.aclose()
    assert o._http.is_closed


# --- timeout paths, against a real local socket (respx returns bodies all at once) ---

HEADERS = b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nConnection: close\r\n\r\n"
TOKEN = ndjson({"message": {"content": "hi"}, "done": False})
DONE = ndjson({"message": {"content": ""}, "done": True, "eval_count": 1, "eval_duration": 10**6})


@pytest.fixture
async def scripted_server():
    """Serve one scripted response per connection: steps are bytes to send or float seconds to sleep."""
    servers, tasks = [], set()

    async def start(steps):
        async def handle(reader, writer):
            tasks.add(asyncio.current_task())
            try:
                head = await reader.readuntil(b"\r\n\r\n")
                length = next((int(h.split(b":")[1]) for h in head.lower().split(b"\r\n")
                               if h.startswith(b"content-length")), 0)
                await reader.readexactly(length)
                for step in steps:
                    if isinstance(step, float):
                        await asyncio.sleep(step)
                    else:
                        writer.write(step)
                        await writer.drain()
            except (ConnectionError, asyncio.IncompleteReadError):
                pass
            finally:
                writer.close()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        servers.append(server)
        return f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"

    yield start
    for t in tasks:
        t.cancel()
    for s in servers:
        s.close()
    await asyncio.gather(*tasks, return_exceptions=True)


async def test_first_token_timeout_covers_slow_headers(scripted_server):
    url = await scripted_server([0.6, HEADERS, TOKEN + DONE])
    async with OllamaClient(url) as c:
        r = await c.chat("m", [], first_token_timeout=0.15, total_timeout=5)
    assert r.error == "first_token_timeout" and r.total_s < 0.5


async def test_first_token_timeout_covers_slow_body(scripted_server):
    url = await scripted_server([HEADERS, 0.6, TOKEN + DONE])
    async with OllamaClient(url) as c:
        r = await c.chat("m", [], first_token_timeout=0.15, total_timeout=5)
    assert r.error == "first_token_timeout" and r.total_s < 0.5


async def test_stall_after_first_token_hits_total_timeout(scripted_server):
    url = await scripted_server([HEADERS, TOKEN, 1.0, DONE])
    async with OllamaClient(url) as c:
        r = await c.chat("m", [], first_token_timeout=0.2, total_timeout=0.4)
    assert r.error == "timeout" and 0.35 < r.total_s < 0.9


async def test_total_timeout_before_first_token_is_not_first_token_timeout(scripted_server):
    url = await scripted_server([0.6, HEADERS, TOKEN + DONE])
    async with OllamaClient(url) as c:
        r = await c.chat("m", [], first_token_timeout=5, total_timeout=0.15)
    assert r.error == "timeout"


async def test_fast_response_passes_both_timeouts(scripted_server):
    url = await scripted_server([HEADERS, TOKEN, DONE])
    async with OllamaClient(url) as c:
        r = await c.chat("m", [], first_token_timeout=1, total_timeout=2)
    assert r.ok and r.content == "hi"


async def test_openai_compat_first_token_timeout_covers_slow_headers(scripted_server):
    url = await scripted_server([0.6, HEADERS, b"data: [DONE]\n\n"])
    async with OpenAICompatClient(url, "k", "m") as c:
        r = await c.chat("hosted", [], first_token_timeout=0.15, total_timeout=5)
    assert r.error == "first_token_timeout"
