import time


class CircuitBreaker:
    """Skip a backend after `threshold` consecutive failures; allow one trial call after `cooldown_s`."""

    def __init__(self, threshold: int = 3, cooldown_s: float = 30.0, clock=time.monotonic):
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.clock = clock
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    def allow(self, name: str) -> bool:
        opened = self._opened_at.get(name)
        return opened is None or self.clock() - opened >= self.cooldown_s

    def record(self, name: str, ok: bool) -> None:
        if ok:
            self._failures.pop(name, None)
            self._opened_at.pop(name, None)
            return
        self._failures[name] = self._failures.get(name, 0) + 1
        if self._failures[name] >= self.threshold:
            self._opened_at[name] = self.clock()
