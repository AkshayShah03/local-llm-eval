import json

import httpx

from evalh.clients.base import AsyncClientMixin, ChatResult, StreamError, stream_chat


class _SSEParser:
    """Parses OpenAI-style chat.completion.chunk SSE lines."""

    def __init__(self):
        self.content, self.slots, self.usage, self.finished = [], {}, {}, False

    def feed(self, line: str) -> bool:
        if not line.startswith("data:"):
            return False
        data = line[5:].strip()
        if data == "[DONE]":
            self.finished = True
            return False
        chunk = json.loads(data)
        if chunk.get("error"):
            err = chunk["error"]
            raise StreamError(f"api: {err.get('message', err) if isinstance(err, dict) else err}")
        self.usage = chunk.get("usage") or self.usage
        got = False
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            got = got or bool(delta.get("content") or delta.get("tool_calls"))
            if delta.get("content"):
                self.content.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                slot = self.slots.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
        return got

    def result(self, total_s: float, ttft_s: float | None) -> ChatResult:
        calls = []
        for _, slot in sorted(self.slots.items()):
            try:
                args = json.loads(slot["arguments"]) if slot["arguments"] else {}
            except json.JSONDecodeError:
                args = slot["arguments"]
            calls.append({"name": slot["name"], "arguments": args})
        completion = self.usage.get("completion_tokens")
        decode = completion / (total_s - ttft_s) if completion and ttft_s is not None and total_s > ttft_s else None
        return ChatResult(
            content="".join(self.content), tool_calls=calls, total_s=total_s, ttft_s=ttft_s,
            prompt_tokens=self.usage.get("prompt_tokens"), completion_tokens=completion,
            cached_prompt_tokens=(self.usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            decode_tps=decode,
        )


class OpenAICompatClient(AsyncClientMixin):
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
        return await stream_chat(self._http, "/chat/completions", body, _SSEParser(),
                                 first_token_timeout, total_timeout)
