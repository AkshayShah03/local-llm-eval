import json

import httpx
import numpy as np

from evalh.clients.base import AsyncClientMixin, ChatResult, StreamError, rate, stream_chat


class _ChatParser:
    """Parses Ollama /api/chat NDJSON lines."""

    def __init__(self):
        self.content, self.calls, self.final, self.finished = [], [], {}, False

    def feed(self, line: str) -> bool:
        if not line.strip():
            return False
        chunk = json.loads(line)
        if "error" in chunk:
            raise StreamError(f"ollama: {chunk['error']}")
        msg = chunk.get("message") or {}
        if msg.get("content"):
            self.content.append(msg["content"])
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            self.calls.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
        if chunk.get("done"):
            self.final, self.finished = chunk, True
        return bool(msg.get("content") or msg.get("tool_calls"))

    def result(self, total_s: float, ttft_s: float | None) -> ChatResult:
        f = self.final
        prompt, cached = f.get("prompt_eval_count"), f.get("prompt_eval_cached_count")
        # prompt_eval_count includes tokens served from the prompt cache (checked live on
        # Ollama 0.40.4), but prompt_eval_duration only covers the new ones.
        return ChatResult(
            content="".join(self.content), tool_calls=self.calls, total_s=total_s, ttft_s=ttft_s,
            prompt_tokens=prompt, completion_tokens=f.get("eval_count"), cached_prompt_tokens=cached,
            prefill_tps=rate((prompt - (cached or 0)) if prompt else None, f.get("prompt_eval_duration")),
            decode_tps=rate(f.get("eval_count"), f.get("eval_duration")),
            load_s=(f["load_duration"] / 1e9) if f.get("load_duration") else None,
        )


class OllamaClient(AsyncClientMixin):
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 120.0):
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def chat(self, model, messages, tools=None, options=None,
                   first_token_timeout: float | None = None,
                   total_timeout: float | None = None) -> ChatResult:
        body = {"model": model, "messages": messages, "stream": True, "options": options or {}}
        if tools:
            body["tools"] = tools
        return await stream_chat(self._http, "/api/chat", body, _ChatParser(),
                                 first_token_timeout, total_timeout)

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
