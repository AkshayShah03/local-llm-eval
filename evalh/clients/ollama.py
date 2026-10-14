import asyncio
import json
import time

import httpx
import numpy as np

from evalh.clients.base import ChatResult, rate


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 120.0):
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def chat(self, model, messages, tools=None, options=None,
                   first_token_timeout: float | None = None,
                   total_timeout: float | None = None) -> ChatResult:
        body = {"model": model, "messages": messages, "stream": True, "options": options or {}}
        if tools:
            body["tools"] = tools
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(total_timeout):
                return await self._stream(body, t0, first_token_timeout)
        except TimeoutError:
            return ChatResult.failed("timeout", time.perf_counter() - t0)
        except httpx.TimeoutException:
            return ChatResult.failed("timeout", time.perf_counter() - t0)
        except httpx.HTTPError as e:
            return ChatResult.failed(f"{type(e).__name__}: {e}", time.perf_counter() - t0)

    async def _stream(self, body, t0, first_token_timeout) -> ChatResult:
        content, calls, final, ttft = [], [], {}, None
        async with self._http.stream("POST", "/api/chat", json=body) as r:
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
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    return ChatResult.failed(f"ollama: {chunk['error']}", time.perf_counter() - t0)
                msg = chunk.get("message") or {}
                if ttft is None and (msg.get("content") or msg.get("tool_calls")):
                    ttft = time.perf_counter() - t0
                if msg.get("content"):
                    content.append(msg["content"])
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    calls.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
                if chunk.get("done"):
                    final = chunk
        return ChatResult(
            content="".join(content), tool_calls=calls, total_s=time.perf_counter() - t0, ttft_s=ttft,
            prompt_tokens=final.get("prompt_eval_count"), completion_tokens=final.get("eval_count"),
            cached_prompt_tokens=final.get("prompt_eval_cached_count"),
            prefill_tps=rate(final.get("prompt_eval_count"), final.get("prompt_eval_duration")),
            decode_tps=rate(final.get("eval_count"), final.get("eval_duration")),
            load_s=(final["load_duration"] / 1e9) if final.get("load_duration") else None,
        )

    async def embed(self, model: str, texts: list[str], batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            r = await self._http.post("/api/embed", json={"model": model, "input": texts[i:i + batch]})
            out += r.raise_for_status().json()["embeddings"]
        return np.array(out, dtype=np.float32)

    async def ps(self) -> list[dict]:
        return (await self._http.get("/api/ps")).raise_for_status().json()["models"]

    async def version(self) -> str:
        return (await self._http.get("/api/version")).raise_for_status().json()["version"]

    async def unload(self, model: str) -> None:
        await self._http.post("/api/generate", json={"model": model, "keep_alive": 0})
