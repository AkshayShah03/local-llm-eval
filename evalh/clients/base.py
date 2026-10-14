import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Protocol

import httpx


@dataclass
class ChatResult:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    total_s: float = 0.0
    ttft_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    prefill_tps: float | None = None
    decode_tps: float | None = None
    load_s: float | None = None
    error: str | None = None

    @classmethod
    def failed(cls, error: str, total_s: float) -> "ChatResult":
        return cls(error=error, total_s=total_s)

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return asdict(self)


def rate(count: int | None, duration_ns: int | None) -> float | None:
    if not count or not duration_ns:
        return None
    return count / (duration_ns / 1e9)


class AsyncClientMixin:
    """Lifecycle for clients that own an httpx.AsyncClient in `self._http`."""

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()


class StreamError(Exception):
    """The server reported an error inside an otherwise successful stream."""


class StreamParser(Protocol):
    finished: bool

    def feed(self, line: str) -> bool:
        """Consume one line. Return True if it carried content or a tool call.
        Raise StreamError for an in-stream error."""

    def result(self, total_s: float, ttft_s: float | None) -> ChatResult: ...


async def stream_chat(http: httpx.AsyncClient, path: str, body: dict, parser: StreamParser,
                      first_token_timeout: float | None, total_timeout: float | None) -> ChatResult:
    """POST body and feed response lines to parser; always returns a ChatResult.

    first_token_timeout covers request start (including the wait for response headers)
    up to the first content or tool-call line. total_timeout covers everything.
    """
    t0 = time.perf_counter()
    total_cm = asyncio.timeout(total_timeout)
    first_cm = asyncio.timeout(first_token_timeout)
    ttft = None
    try:
        async with total_cm, first_cm:
            async with http.stream("POST", path, json=body) as r:
                if r.status_code >= 400:
                    text = (await r.aread()).decode(errors="replace")
                    return ChatResult.failed(f"http {r.status_code}: {text[:200]}", time.perf_counter() - t0)
                async for line in r.aiter_lines():
                    if parser.feed(line) and ttft is None:
                        ttft = time.perf_counter() - t0
                        first_cm.reschedule(None)
                    if parser.finished:
                        break
        return parser.result(time.perf_counter() - t0, ttft)
    except StreamError as e:
        return ChatResult.failed(str(e), time.perf_counter() - t0)
    except TimeoutError:
        # Only asyncio.timeout raises this; expired() says which deadline it was.
        kind = "first_token_timeout" if first_cm.expired() and not total_cm.expired() else "timeout"
        return ChatResult.failed(kind, time.perf_counter() - t0)
    except httpx.TimeoutException:
        return ChatResult.failed("timeout", time.perf_counter() - t0)
    except Exception as e:  # json.JSONDecodeError, httpx.StreamError, anything else: never raise
        return ChatResult.failed(f"{type(e).__name__}: {e}", time.perf_counter() - t0)
