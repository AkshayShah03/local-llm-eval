import asyncio
import json
import time

import httpx

from evalh.clients.base import ChatResult


class OpenAICompatClient:
    """Any /chat/completions endpoint: OpenAI, Groq, Together, vLLM."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0):
        self.model = model
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout,
                                       headers={"Authorization": f"Bearer {api_key}"})

    async def chat(self, model, messages, tools=None, options=None,
                   first_token_timeout: float | None = None,
                   total_timeout: float | None = None) -> ChatResult:
        opts = options or {}
        body = {"model": self.model, "messages": messages, "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": opts.get("temperature", 0)}
        if "seed" in opts:
            body["seed"] = opts["seed"]
        if "num_predict" in opts:
            body["max_tokens"] = opts["num_predict"]
        if tools:
            body["tools"] = tools
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(total_timeout):
                return await self._stream(body, t0, first_token_timeout)
        except (TimeoutError, httpx.TimeoutException):
            return ChatResult.failed("timeout", time.perf_counter() - t0)
        except httpx.HTTPError as e:
            return ChatResult.failed(f"{type(e).__name__}: {e}", time.perf_counter() - t0)

    async def _stream(self, body, t0, first_token_timeout) -> ChatResult:
        content, slots, usage, ttft = [], {}, {}, None
        async with self._http.stream("POST", "/chat/completions", json=body) as r:
            if r.status_code >= 400:
                text = (await r.aread()).decode(errors="replace")
                return ChatResult.failed(f"http {r.status_code}: {text[:200]}", time.perf_counter() - t0)
            lines = r.aiter_lines()
            while True:
                try:
                    if ttft is None and first_token_timeout is not None:
                        remaining = first_token_timeout - (time.perf_counter() - t0)
                        line = await asyncio.wait_for(anext(lines), max(remaining, 0.001))
                    else:
                        line = await anext(lines)
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    return ChatResult.failed("first_token_timeout", time.perf_counter() - t0)
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    if ttft is None and (delta.get("content") or delta.get("tool_calls")):
                        ttft = time.perf_counter() - t0
                    if delta.get("content"):
                        content.append(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        slot = slots.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
        total = time.perf_counter() - t0
        calls = []
        for _, slot in sorted(slots.items()):
            try:
                args = json.loads(slot["arguments"]) if slot["arguments"] else {}
            except json.JSONDecodeError:
                args = slot["arguments"]
            calls.append({"name": slot["name"], "arguments": args})
        completion = usage.get("completion_tokens")
        decode = completion / (total - ttft) if completion and ttft is not None and total > ttft else None
        return ChatResult(
            content="".join(content), tool_calls=calls, total_s=total, ttft_s=ttft,
            prompt_tokens=usage.get("prompt_tokens"), completion_tokens=completion,
            cached_prompt_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            decode_tps=decode,
        )
