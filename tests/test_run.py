from evalh.run import done_keys, is_outage, order_items, undone_keys


def test_done_keys_only_counts_responses():
    recs = [{"type": "run_meta"},
            {"type": "response", "config": "m", "task": "rag", "item_id": "rag-001", "repeat": 1}]
    assert done_keys(recs) == {("m", "rag", "rag-001", 1)}


def test_order_items_is_seeded_and_differs_per_repeat():
    items = [{"id": str(i)} for i in range(20)]
    a1, a2 = order_items(items, repeat=1), order_items(items, repeat=2)
    assert a1 == order_items(items, repeat=1)
    assert [x["id"] for x in a1] != [x["id"] for x in a2]
    assert sorted(x["id"] for x in a1) == sorted(x["id"] for x in items)


def test_is_outage_only_transport_failures():
    for e in ("ConnectError: refused", "ReadError: x", "RemoteProtocolError: eof", "WriteError: y",
              "http 502: bad gateway", "http 503: down", "http 504: t/o"):
        assert is_outage(e), e
    for e in (None, "timeout", "first_token_timeout", "http 400: x", "http 404: x", "http 500: boom",
              "ollama: model runner crashed", "api: bad", "JSONDecodeError: x", "KeyError: 'x'"):
        assert not is_outage(e), e


# ---- run_model with a fake client ----
import pytest

from evalh.clients.base import ChatResult
from evalh.config import ModelConfig
from evalh.data import read_jsonl
from evalh.run import RunAborted, WarmupFailed, run_model

MODEL = ModelConfig(name="m", family="f", params_b=1, quant="q")
DEFAULTS = {"temperature": 0, "seed": 7, "num_ctx": 8192, "num_predict": 512}


class FakeClient:
    def __init__(self, errors=None, warm_error=None):
        self.warm_error = warm_error
        self.errors = errors or {}  # call index (after warmup) -> error string
        self.calls = 0
        self.unloaded = []

    async def chat(self, model, messages, tools=None, options=None, **kw):
        if options.get("num_predict") == 1:
            return ChatResult(error=self.warm_error) if self.warm_error else ChatResult(content="ok", load_s=0.5)
        self.calls += 1
        err = self.errors.get(self.calls, self.errors.get("all"))
        return ChatResult(error=err, total_s=0.1) if err else ChatResult(content="x", total_s=0.1)

    async def ps(self):
        return [{"name": "m", "digest": "d", "size": 10, "size_vram": 10, "context_length": 8192}]

    async def unload(self, model):
        self.unloaded.append(model)


def _items(n=8):
    return {"triage": [{"id": f"t{i}", "log": "l"} for i in range(n)]}


async def _run(tmp_path, client, done=None, items=None, **kw):
    out = tmp_path / "raw.jsonl"
    await run_model(MODEL, client, client, ["triage"], items or _items(), None, {}, DEFAULTS, out,
                    done or set(), "r", 1, errors_path=tmp_path / "err.jsonl", **kw)
    return read_jsonl(out)


async def test_prefilled_done_keys_skipped_and_meta_partial(tmp_path):
    done = {("m", "triage", f"t{i}", 1) for i in range(3)}
    client = FakeClient()
    recs = await _run(tmp_path, client, done=done)
    resp = [r for r in recs if r["type"] == "response"]
    assert len(resp) == 5 and client.calls == 5
    assert not {r["item_id"] for r in resp} & {"t0", "t1", "t2"}
    assert sum(r["type"] == "model_load" for r in recs) == 1
    meta = [r for r in recs if r["type"] == "model_meta"]
    assert len(meta) == 1 and meta[0]["partial"] is True and meta[0]["responses_in_window"] == 5


async def test_fresh_run_not_partial_and_load_record(tmp_path):
    recs = await _run(tmp_path, FakeClient())
    load = next(r for r in recs if r["type"] == "model_load")
    assert load["digest"] == "d" and load["size_vram"] == 10 and "system_used_delta_after_load" in load
    meta = next(r for r in recs if r["type"] == "model_meta")
    assert meta["partial"] is False and meta["responses_in_window"] == 8


async def test_timeouts_and_http_4xx_are_recorded(tmp_path):
    client = FakeClient(errors={1: "timeout", 2: "first_token_timeout", 3: "http 400: bad"})
    recs = await _run(tmp_path, client)
    resp = [r for r in recs if r["type"] == "response"]
    assert len(resp) == 8
    assert sorted(r["result"]["error"] for r in resp if r["result"]["error"]) == [
        "first_token_timeout", "http 400: bad", "timeout"]


async def test_unexpected_errors_not_recorded_and_abort_after_five(tmp_path):
    client = FakeClient(errors={"all": "ConnectError: refused"})
    with pytest.raises(RunAborted):
        await _run(tmp_path, client)
    recs = read_jsonl(tmp_path / "raw.jsonl")
    assert not [r for r in recs if r["type"] == "response"]
    assert not [r for r in recs if r["type"] == "model_meta"]
    assert len(read_jsonl(tmp_path / "err.jsonl")) == 5


async def test_isolated_unexpected_error_does_not_abort_and_is_not_done(tmp_path):
    client = FakeClient(errors={2: "ConnectError: blip"})
    recs = await _run(tmp_path, client)
    assert len([r for r in recs if r["type"] == "response"]) == 7
    assert len(read_jsonl(tmp_path / "err.jsonl")) == 1


def test_repeat_order_does_not_start_with_previous_last():
    items = [{"id": str(i)} for i in range(2)]
    for r in range(2, 12):
        assert order_items(items, r)[0] != order_items(items, r - 1)[-1]


async def test_deterministic_errors_are_recorded_as_responses(tmp_path):
    client = FakeClient(errors={"all": "ollama: model runner has unexpectedly stopped"})
    recs = await _run(tmp_path, client)
    resp = [r for r in recs if r["type"] == "response"]
    assert len(resp) == 8 and all(r["result"]["error"] for r in resp)
    assert not (tmp_path / "err.jsonl").exists()


async def test_warmup_failure_writes_load_error_and_raises_warmup_failed(tmp_path):
    client = FakeClient(warm_error="ollama: model requires more system memory")
    with pytest.raises(WarmupFailed):
        await _run(tmp_path, client)
    recs = read_jsonl(tmp_path / "raw.jsonl")
    assert [r["type"] for r in recs] == ["model_load"] and "memory" in recs[0]["error"]
    assert client.calls == 0 and client.unloaded == ["m"]


async def test_warmup_outage_aborts(tmp_path):
    with pytest.raises(RunAborted):
        await _run(tmp_path, FakeClient(warm_error="ConnectError: refused"))


def test_undone_keys():
    items = _items(2)
    done = {("m", "triage", "t0", 1)}
    assert undone_keys([MODEL], ["triage"], items, 2, done) == [
        ("m", "triage", "t1", 1), ("m", "triage", "t0", 2), ("m", "triage", "t1", 2)]


async def test_ollama_client_tags():
    import respx
    from httpx import Response

    from evalh.clients.ollama import OllamaClient
    with respx.mock(base_url="http://localhost:11434") as m:
        m.get("/api/tags").respond(json={"models": [{"name": "a:latest", "digest": "d1"}]})
        async with OllamaClient() as c:
            assert await c.tags() == [{"name": "a:latest", "digest": "d1"}]
