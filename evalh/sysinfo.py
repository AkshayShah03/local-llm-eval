import platform
import subprocess
import sys
import threading
import time

import psutil


def _sysctl(key: str) -> str:
    return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()


def machine_info() -> dict:
    return {
        "chip": _sysctl("machdep.cpu.brand_string") or platform.processor(),
        "memory_gb": round(int(_sysctl("hw.memsize") or 0) / 2**30, 1),
        "os": f"macOS {platform.mac_ver()[0]}" if platform.system() == "Darwin" else platform.platform(),
        "python": platform.python_version(),
    }


def _ollama_procs():
    for p in psutil.process_iter(["name", "cmdline"]):
        if (p.info["name"] or "").startswith("ollama"):
            yield p


def is_runner(name: str, cmdline: list[str]) -> bool:
    """The model-serving child of `ollama serve`: `ollama runner` in older releases, `llama-server` in newer ones."""
    return name.startswith("llama-server") or "runner" in cmdline or "llama-server" in (cmdline[0] if cmdline else "")


def _find_serve():
    for p in _ollama_procs():
        if "serve" in (p.info["cmdline"] or []):
            return p
    return None


def _runner_procs(serve=None):
    """Runner children of `ollama serve`. Pass a cached `serve` process to skip the process-table scan."""
    serve = serve or _find_serve()
    if serve is None:
        return
    try:
        children = serve.children(recursive=True)
    except psutil.Error:
        return
    for c in children:
        try:
            if is_runner(c.name(), c.cmdline()):
                yield c
        except psutil.Error:
            pass


def ollama_env() -> dict:
    """OLLAMA_* variables of the running server; these change memory use (KV cache type, flash attention)."""
    for p in _ollama_procs():
        if "serve" in (p.info["cmdline"] or []):
            try:
                return {k: v for k, v in p.environ().items() if k.startswith("OLLAMA_")}
            except psutil.AccessDenied:
                return {"error": "access denied"}
    return {}


def git_state(cwd=None) -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=cwd).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                cwd=cwd).stdout.strip())
    return {"sha": sha, "dirty": dirty}


def system_used() -> int:
    vm = psutil.virtual_memory()
    return vm.total - vm.available


class MemorySampler:
    """Samples peak RSS of Ollama runner processes and system memory in a background thread.

    On Apple Silicon, Metal buffers are not fully reflected in process RSS, so the
    system-wide delta is the more honest number; both are recorded.
    """

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak_runner_rss = 0
        self.peak_system_used = 0
        self.baseline_system_used = system_used()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        serve = None
        while not self._stop.is_set():
            try:
                if serve is None or not serve.is_running():
                    serve = _find_serve()
                rss = 0
                for p in _runner_procs(serve):
                    try:
                        rss += p.memory_info().rss
                    except psutil.Error:
                        pass
                self.peak_runner_rss = max(self.peak_runner_rss, rss)
                self.peak_system_used = max(self.peak_system_used, system_used())
            except Exception as e:  # keep sampling; a silent dead thread would record zeros as measurements
                print(f"memory sampler error: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
