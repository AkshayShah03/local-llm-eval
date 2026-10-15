from evalh.run import done_keys, is_unexpected_error, order_items


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


def test_is_unexpected_error():
    assert not is_unexpected_error(None)
    assert not is_unexpected_error("timeout")
    assert not is_unexpected_error("first_token_timeout")
    assert not is_unexpected_error("http 404: nope")
    assert is_unexpected_error("http 500: boom")
    assert is_unexpected_error("http 503: down")
    assert is_unexpected_error("KeyError: 'x'")


# ---- run_model with a fake client ----
import pytest

from evalh.clients.base import ChatResult
from evalh.config import ModelConfig
from evalh.data import read_jsonl
from evalh.run import RunAborted, run_model

MODEL = ModelConfig(name="m", family="f", params_b=1, quant="q")
DEFAULTS = {"temperature": 0, "seed": 7, "num_ctx": 8192, "num_predict": 512}


class FakeClient:
    def __init__(self, errors=None):
        self.errors = errors or {}  # call index (after warmup) -> error string
        self.calls = 0
        self.unloaded = []

    async def chat(self, model, messages, tools=None, options=None, **kw):
        if options.get("num_predict") == 1:
            return ChatResult(content="ok", load_s=0.5)
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
    client = FakeClient(errors={"all": "http 503: down"})
    with pytest.raises(RunAborted):
        await _run(tmp_path, client)
    recs = read_jsonl(tmp_path / "raw.jsonl")
    assert not [r for r in recs if r["type"] == "response"]
    assert not [r for r in recs if r["type"] == "model_meta"]
    assert len(read_jsonl(tmp_path / "err.jsonl")) == 5


async def test_isolated_unexpected_error_does_not_abort_and_is_not_done(tmp_path):
    client = FakeClient(errors={2: "KeyError: 'x'"})
    recs = await _run(tmp_path, client)
    assert len([r for r in recs if r["type"] == "response"]) == 7
    assert len(read_jsonl(tmp_path / "err.jsonl")) == 1


def test_repeat_order_does_not_start_with_previous_last():
    items = [{"id": str(i)} for i in range(2)]
    for r in range(2, 12):
        assert order_items(items, r)[0] != order_items(items, r - 1)[-1]
