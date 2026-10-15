import asyncio
import json

import numpy as np

from evalh.data import read_jsonl
from evalh.rag.index import Index
from evalh.score import deterministic_scores, score_run


def rec(content="", tool_calls=None, error=None, retrieved=None):
    return {"retrieved": retrieved, "result": {"content": content, "tool_calls": tool_calls or [], "error": error}}


def test_rag_scores():
    item = {"answerable": True, "required_facts": ["65535:666"], "gold_chunk_ids": ["rfc7999-001"]}
    s = deterministic_scores("rag", item, rec("Use 65535:666 [rfc7999-001]", retrieved=["x", "rfc7999-001"]))
    assert s == {"failed": False, "fact_recall": 1.0, "refused": False, "answerable": True, "retrieval_hit": True}


def test_unanswerable_refusal():
    item = {"answerable": False, "required_facts": [], "gold_chunk_ids": []}
    s = deterministic_scores("rag", item, rec("I can't answer that from the provided documents.", retrieved=["a"]))
    assert s["refused"] and s["retrieval_hit"] is None


def test_tool_scores():
    item = {"expected_calls": [{"name": "get_bgp_session", "args": {"neighbor": "192.0.2.1"}}]}
    s = deterministic_scores("tools", item, rec(tool_calls=[{"name": "get_bgp_session", "arguments": {"neighbor": "192.0.2.1"}}]))
    assert s["selection_ok"] and s["args_ok"] and s["expects_tool"] and not s["invalid"]


def test_failed_response():
    s = deterministic_scores("tools", {"expected_calls": []}, rec(error="timeout"))
    assert s == {"failed": True}


class FakeResult:
    def __init__(self, content, ok=True):
        self.content, self.ok = content, ok


class FakeJudge:
    def __init__(self, replies):
        self.replies, self.calls, self.closed = list(replies), 0, False

    async def chat(self, model, messages, tools=None, options=None):
        self.calls += 1
        r = self.replies.pop(0)
        return FakeResult(r, ok=r is not None) if r is not None else FakeResult("", ok=False)

    async def aclose(self):
        self.closed = True


GOOD = json.dumps({"correctness": 4, "faithfulness": 5, "reason": "ok"})


def response(config, task, item_id, repeat, **kw):
    return {"type": "response", "config": config, "task": task, "item_id": item_id, "repeat": repeat, **rec(**kw)}


def setup(tmp_path):
    chunks = [{"id": "c1", "text": "Use 65535:666 to blackhole."}]
    index = Index(chunks, np.array([[1.0, 0.0]]))
    items = {
        "r1": ("rag", {"id": "r1", "question": "q?", "answerable": True, "required_facts": ["65535:666"],
                       "gold_chunk_ids": ["c1"]}),
        "r2": ("rag", {"id": "r2", "question": "q2?", "answerable": True, "required_facts": ["x"],
                       "gold_chunk_ids": ["c1"]}),
        "t1": ("tools", {"id": "t1", "expected_calls": []}),
        "g1": ("triage", {"id": "g1", "log": "log text", "required_facts": ["down"]}),
    }
    raw = [
        {"type": "run_meta", "run_id": "x"},
        response("m", "rag", "r1", 1, content="65535:666", retrieved=["c1"]),
        response("m", "rag", "r1", 2, content="65535:666", retrieved=["c1"]),
        response("m", "rag", "r2", 1, content="nope", retrieved=["c1"]),
        response("m", "tools", "t1", 1, content="hi"),
        response("m", "triage", "g1", 1, content="link down"),
        response("m", "triage", "g1", 1, error="timeout"),
        {"type": "model_meta", "config": "m"},
    ]
    return raw, items, index, tmp_path / "cache.jsonl", tmp_path / "out" / "scored.jsonl"


def test_score_run_end_to_end(tmp_path):
    raw, items, index, cache, out = setup(tmp_path)
    jc = FakeJudge([GOOD, None, GOOD])  # r1, r2 (fails), g1
    asyncio.run(score_run(raw, items, index, jc, "fake", cache, out))
    rows = read_jsonl(out)
    assert len(rows) == len(raw)
    assert rows[0] == raw[0] and rows[-1] == raw[-1]
    by = {(r["item_id"], r["repeat"], r["scores"]["failed"]): r for r in rows if r["type"] == "response"}
    r1 = by[("r1", 1, False)]["scores"]
    assert r1["correctness"] == 4 and r1["faithfulness"] == 5 and r1["fact_recall"] == 1.0
    assert "judge_failed" not in r1
    assert "correctness" not in by[("r1", 2, False)]["scores"]  # only repeat 1 judged
    r2 = by[("r2", 1, False)]["scores"]
    assert r2["judge_failed"] is True and "correctness" not in r2
    assert by[("t1", 1, False)]["scores"]["expects_tool"] is False
    assert by[("g1", 1, False)]["scores"]["correctness"] == 4
    assert by[("g1", 1, True)]["scores"] == {"failed": True}
    assert jc.calls == 3
    assert all(r["judge"] == "fake" for r in rows if r["type"] == "response")
    # failures are not cached
    assert {c["item_id"] for c in read_jsonl(cache)} == {"r1", "g1"}


def test_rerun_uses_cache_and_retries_failures(tmp_path):
    raw, items, index, cache, out = setup(tmp_path)
    asyncio.run(score_run(raw, items, index, FakeJudge([GOOD, None, GOOD]), "fake", cache, out))
    jc = FakeJudge([GOOD])
    asyncio.run(score_run(raw, items, index, jc, "fake", cache, out))
    assert jc.calls == 1  # only the previously failed r2
    rows = [r for r in read_jsonl(out) if r["type"] == "response"]
    assert not any(r["scores"].get("judge_failed") for r in rows)
    assert len(rows) == 6


def test_main_requires_hosted_key(monkeypatch, capsys):
    import pytest
    from evalh import score
    monkeypatch.delenv("HOSTED_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["score", "--run-id", "x", "--judge", "hosted"])
    with pytest.raises(SystemExit) as e:
        asyncio.run(score.main())
    assert e.value.code != 0
    assert "HOSTED_API_KEY" in capsys.readouterr().err


def test_cache_hit_on_unchanged_input(tmp_path):
    raw, items, index, cache, out = setup(tmp_path)
    asyncio.run(score_run(raw, items, index, FakeJudge([GOOD, GOOD, GOOD]), "m1", cache, out))
    jc = FakeJudge([])
    asyncio.run(score_run(raw, items, index, jc, "m1", cache, out))
    assert jc.calls == 0


def test_changed_judge_model_rejudges(tmp_path):
    raw, items, index, cache, out = setup(tmp_path)
    asyncio.run(score_run(raw, items, index, FakeJudge([GOOD, GOOD, GOOD]), "m1", cache, out))
    jc = FakeJudge([GOOD, GOOD, GOOD])
    asyncio.run(score_run(raw, items, index, jc, "m2", cache, out))
    assert jc.calls == 3


def test_changed_answer_rejudges(tmp_path):
    raw, items, index, cache, out = setup(tmp_path)
    asyncio.run(score_run(raw, items, index, FakeJudge([GOOD, GOOD, GOOD]), "m1", cache, out))
    raw[1] = response("m", "rag", "r1", 1, content="a different answer", retrieved=["c1"])
    jc = FakeJudge([GOOD])
    asyncio.run(score_run(raw, items, index, jc, "m1", cache, out))
    assert jc.calls == 1


def test_changed_required_facts_rejudges(tmp_path):
    raw, items, index, cache, out = setup(tmp_path)
    asyncio.run(score_run(raw, items, index, FakeJudge([GOOD, GOOD, GOOD]), "m1", cache, out))
    items["r1"][1]["required_facts"] = ["65535:666", "extra"]
    jc = FakeJudge([GOOD])
    asyncio.run(score_run(raw, items, index, jc, "m1", cache, out))
    assert jc.calls == 1
