from dataclasses import asdict, dataclass, field


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
