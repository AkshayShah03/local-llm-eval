import json
import math
import tempfile
from pathlib import Path

import yaml

from evalh.report import (_charts, aggregate, effective_loads, evaluate, fmt, memory_series, recommend, render,
                          smaller_unevaluated, throughput_series)


def _write_questions():
    d = Path(tempfile.mkdtemp())
    rows = lambda items: "\n".join(json.dumps(i) for i in items) + "\n"  # noqa: E731
    (d / "rag.jsonl").write_text(rows([{"id": i, "answerable": a} for i, a in
                                       [("a", True), ("b", True), ("c", True), ("d", True), ("u", False)]]))
    tools = [{"id": f"{p}{i}", "expected_calls": [{"name": "f"}]} for p in "tu" for i in range(10)]
    tools += [{"id": "x", "expected_calls": [{"name": "f"}]}, {"id": "nt", "expected_calls": []}]
    (d / "tools.jsonl").write_text(rows(tools))
    return d


QDIR = _write_questions()


def agg(rows):
    return aggregate(rows, QDIR)


def resp(config, task, item, repeat, total, scores, decode=20.0, prefill=None, prompt=None, cached=0):
    return {"type": "response", "config": config, "task": task, "item_id": item, "repeat": repeat,
            "result": {"total_s": total, "ttft_s": 0.2, "decode_tps": decode, "prefill_tps": prefill,
                       "prompt_tokens": prompt, "cached_prompt_tokens": cached}, "scores": scores}


def tool_rows(config, sel, latency):
    rows = []
    for i in range(10):
        ok = i < sel
        for rep in (1, 2):
            rows.append(resp(config, "tools", f"t{i}", rep, latency,
                             {"failed": False, "selection_ok": ok, "args_ok": ok, "invalid": False, "expects_tool": True}))
    return rows


def test_aggregate_tools():
    m = agg(tool_rows("small", 9, 1.0))["small"]["tools"]
    assert m["tool_selection"] == 0.9 and m["n"] == 20 and m["latency_p95_s"] == 1.0
    assert m["n_metrics"]["tool_selection"] == 10


def test_evaluate_and_recommend():
    metrics = agg(tool_rows("small", 9, 1.0) + tool_rows("big", 10, 2.0))
    acc = {"noc_tools": {"task": "tools", "criteria": {"tool_selection": {"min": 0.95}, "latency_p95_s": {"max": 4.0}}}}
    ev = evaluate(acc, metrics)
    assert ev["noc_tools"]["small"] == {"pass": False, "failed": ["tool_selection"]}
    assert ev["noc_tools"]["big"]["pass"] is True
    assert recommend(ev, {"small": 1, "big": 5}) == {"noc_tools": "big"}


def test_missing_metric_is_not_evaluated():
    metrics = agg(tool_rows("small", 9, 1.0))
    acc = {"x": {"task": "tools", "criteria": {"faithfulness": {"min": 4}}}}
    assert evaluate(acc, metrics)["x"]["small"]["pass"] is None


def test_recommend_without_hosted_and_with_hosted():
    ev = {"u": {"a": {"pass": False, "failed": ["x"]}}}
    assert recommend(ev, {"a": 1}) == {"u": "none"}
    ev = {"u": {"a": {"pass": False, "failed": ["x"]}, "hosted": {"pass": True, "failed": []}}}
    assert recommend(ev, {"a": 1}) == {"u": "hosted"}


def test_failed_response_counts_in_failure_rate_not_latency():
    rows = tool_rows("m", 10, 1.0)
    rows.append(resp("m", "tools", "t0", 3, 9.0, {"failed": True}))
    m = agg(rows)["m"]["tools"]
    assert m["n"] == 21 and abs(m["failure_rate"] - 1 / 21) < 1e-9 and m["latency_p95_s"] == 1.0


def rag_row(config, item, answerable, recall, refused, judge=None, failed_judge=False):
    s = {"failed": False, "fact_recall": recall, "refused": refused, "answerable": answerable,
         "retrieval_hit": True if answerable else None}
    if judge:
        s.update(judge)
    if failed_judge:
        s["judge_failed"] = True
    return resp(config, "rag", item, 1, 1.0, s)


def test_judge_failed_excluded_and_counted():
    rows = [rag_row("m", "a", True, 1.0, False, {"correctness": 5, "faithfulness": 4}),
            rag_row("m", "b", True, 0.5, False, failed_judge=True),
            rag_row("m", "c", True, 0.0, False, {"correctness": 3}),
            rag_row("m", "d", False, None, True)]
    m = agg(rows)["m"]["rag"]
    assert m["faithfulness"] == 4 and m["n_metrics"]["faithfulness"] == 1
    assert m["correctness"] == 4 and m["n_metrics"]["correctness"] == 2
    assert m["judged"] == 2 and m["judge_failed"] == 1
    assert m["fact_recall"] == 0.5 and m["refusal_unanswerable"] == 1.0 and m["false_refusal"] == 0.0


def triage_row(config, item, correctness):
    return resp(config, "triage", item, 1, 1.0, {"failed": False, "fact_recall": 1.0, "correctness": correctness})


def test_quality_vs_baseline_with_and_without_hosted():
    both = agg([triage_row("m", "a", 4), triage_row("hosted", "a", 5)])
    assert abs(both["m"]["triage"]["quality_vs_baseline"] - (-0.2)) < 1e-9
    local = agg([triage_row("m", "a", 4)])
    assert local["m"]["triage"]["quality_vs_baseline"] is None
    acc = {"t": {"task": "triage", "criteria": {"quality_vs_baseline": {"min": -0.1}}}}
    assert evaluate(acc, local)["t"]["m"]["pass"] is None


def test_prefill_median_ignores_small_fresh_counts():
    rg = {"failed": False, "answerable": False, "refused": True}
    tl = {"failed": False, "expects_tool": False, "selection_ok": True, "invalid": False}
    rows = [resp("m", "rag", "a", 1, 1, rg, prefill=100.0, prompt=500, cached=0),
            resp("m", "rag", "b", 1, 1, rg, prefill=300.0, prompt=500, cached=100),
            resp("m", "rag", "c", 1, 1, rg, prefill=9999.0, prompt=100, cached=60),
            resp("m", "tools", "d", 1, 1, tl, prefill=500.0, prompt=700, cached=0)]
    m = agg(rows)["m"]
    assert m["rag"]["prefill_tps_median"] == 200.0 and m["rag"]["n_metrics"]["prefill_tps_median"] == 2
    assert throughput_series(rows)["m"]["prefill_tps_median"] == 300.0


def test_effective_loads_and_memory_series():
    recs = [{"type": "model_load", "config": "a", "size_vram": 1}, {"type": "model_load", "config": "a", "size_vram": 2},
            {"type": "model_load", "config": "b", "error": "boom"},
            {"type": "model_load", "config": "c", "size_vram": 5}, {"type": "model_load", "config": "c", "error": "x"}]
    loads, status = effective_loads(recs)
    assert loads["a"]["size_vram"] == 2 and loads["c"]["size_vram"] == 5
    assert status == {"a": "ok", "b": "failed", "c": "reload failed"}
    assert memory_series(["a", "b", "c", "hosted"], loads) == [("a", 2 / 1e9), ("c", 5 / 1e9)]


def test_reload_rows_excluded_from_latency():
    rows = tool_rows("m", 10, 1.0)
    rows.append(resp("m", "tools", "t0", 3, 50.0, {"failed": False, "selection_ok": True, "args_ok": True,
                                                    "invalid": False, "expects_tool": True}))
    rows[-1]["result"]["load_s"] = 30.0
    m = agg(rows)["m"]["tools"]
    assert m["latency_p95_s"] == 1.0 and m["reloads_excluded"] == 1 and m["n_metrics"]["latency_p95_s"] == 20


def timeout_row(config, item, total=120.0):
    r = resp(config, "tools", item, 1, total, {"failed": True})
    r["result"]["error"] = "timeout"
    return r


def test_timeouts_count_as_wrong_and_slow():
    rows = []
    for i in range(5):
        rows.append(resp("m", "tools", f"t{i}", 1, 1.0, {"failed": False, "selection_ok": True, "args_ok": True,
                                                         "invalid": False, "expects_tool": True}))
        rows.append(timeout_row("m", f"u{i}"))
    metrics = agg(rows)
    m = metrics["m"]["tools"]
    assert m["tool_selection"] == 0.5 and m["args_exact"] == 0.5 and m["invalid_rate"] == 0.0
    assert m["n_metrics"]["invalid_rate"] == 5 and m["latency_p95_s"] == 120.0
    acc = {"noc_tools": {"task": "tools", "criteria": {"tool_selection": {"min": 0.9}, "latency_p95_s": {"max": 4.0}}}}
    assert evaluate(acc, metrics)["noc_tools"]["m"] == {"pass": False, "failed": ["tool_selection", "latency_p95_s"]}


def test_non_timeout_failure_excluded_from_latency_but_wrong():
    row = resp("m", "tools", "x", 1, 99.0, {"failed": True})
    row["result"]["error"] = "http 400"
    m = agg(tool_rows("m", 10, 1.0) + [row])["m"]["tools"]
    assert m["latency_p95_s"] == 1.0 and abs(m["tool_selection"] - 10 / 11) < 1e-9


def test_failed_rag_unanswerable_refusal_false_and_judge_missing():
    ok_unans = rag_row("o", "u", False, None, True)
    failed = resp("m", "rag", "u", 1, 1.0, {"failed": True})
    failed["result"]["error"] = "timeout"
    m = agg([ok_unans, failed, rag_row("m", "a", True, 1.0, False, {"correctness": 5, "faithfulness": 5})])["m"]["rag"]
    assert m["refusal_unanswerable"] == 0.0 and m["judge_missing_failed_response"] == 0 and m["judged"] == 1
    tri = resp("m", "triage", "t", 1, 1.0, {"failed": True})
    mt = agg([tri])["m"]["triage"]
    assert mt["fact_recall"] == 0.0 and mt["correctness"] == 1.0 and mt["judge_missing_failed_response"] == 1


def test_item_metadata_exact_when_every_config_failed_it():
    def fail(config, task, item):
        r = resp(config, task, item, 1, 1.0, {"failed": True})
        r["result"]["error"] = "timeout"
        return r
    rows = [fail("m", "rag", "u"), fail("n", "rag", "u"), fail("m", "tools", "nt"), fail("n", "tools", "nt")]
    metrics = agg(rows)
    assert metrics["m"]["rag"]["refusal_unanswerable"] == 0.0 and metrics["m"]["rag"]["n_metrics"]["fact_recall"] == 0
    t = metrics["m"]["tools"]
    assert t["abstain_ok"] == 0.0 and t["n_metrics"]["tool_selection"] == 0 and t["n_metrics"]["invalid_rate"] == 0


def test_quality_vs_baseline_is_paired():
    rows = [triage_row("m", "a", 4), triage_row("m", "b", 5), triage_row("hosted", "a", 5)]
    m = agg(rows)["m"]["triage"]
    assert abs(m["quality_vs_baseline"] - (-0.2)) < 1e-9 and m["n_metrics"]["quality_vs_baseline"] == 1


def test_smaller_unevaluated_noted():
    ev = {"u": {"a": {"pass": None, "failed": []}, "b": {"pass": True, "failed": []},
                "c": {"pass": None, "failed": []}}}
    assert smaller_unevaluated(ev, {"a": 1, "b": 2, "c": 3}, "u", "b") == ["a"]
    assert smaller_unevaluated(ev, {}, "u", "none") == []


def test_charts_write_pngs(tmp_path):
    rows = tool_rows("a-instruct", 9, 1.0)
    _charts(["a-instruct"], agg(rows), {"a-instruct": {"size_vram": 2e9}}, throughput_series(rows), tmp_path)
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["memory.png", "quality_vs_latency.png", "throughput.png"]


def test_nan_is_undefined():
    assert fmt(math.nan) == "undefined" and fmt(None) == "–"


def test_render_end_to_end(tmp_path):
    scored = [
        {"type": "run_meta", "run_id": "r", "machine": {"chip": "M4", "memory_gb": 16, "os": "macOS"},
         "ollama_version": "0.40.4", "ollama_env": {"OLLAMA_FLASH_ATTENTION": "1"},
         "git": {"sha": "a" * 40, "dirty": False}},
        {"type": "invocation", "git": {"sha": "a" * 40, "dirty": False}},
        {"type": "invocation", "git": {"sha": "b" * 40, "dirty": True}},
        {"type": "model_load", "config": "small", "size_vram": 2_000_000_000, "warmup_load_s": 1.5},
        {"type": "model_load", "config": "broken", "error": "load failed"},
        {"type": "model_meta", "config": "small", "partial": True, "peak_runner_rss": 1, "system_used_delta": 1},
        {"type": "model_meta", "config": "small", "partial": False, "peak_runner_rss": 3_000_000_000,
         "system_used_delta": 2_500_000_000},
        {"type": "model_meta", "config": "broken", "partial": False},
        *[{**r, "judge": "judge-x"} for r in tool_rows("small", 10, 1.0)],
        *[{**rag_row("small", "a", True, 1.0, False, {"correctness": 5, "faithfulness": 5}), "judge": "judge-x"},
          {**rag_row("small", "b", True, 1.0, False, failed_judge=True), "judge": "judge-x"}],
        triage_row("small", "a", 4),
        {**triage_row("small", "n", 4), "scores": {"failed": False, "fact_recall": float("nan"), "correctness": 4}},
    ]
    results = tmp_path / "results"
    results.mkdir()
    (results / "scored-r.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in scored) + "\n")
    acc = tmp_path / "acc.yaml"
    acc.write_text(yaml.safe_dump({"use_cases": {
        "noc_tools": {"task": "tools", "criteria": {"tool_selection": {"min": 0.9}}},
        "triage": {"task": "triage", "criteria": {"quality_vs_baseline": {"min": -0.1}}}}}))
    out = tmp_path / "reports"
    render("r", results_dir=results, out_dir=out, acceptance_path=acc, charts=False, questions_dir=QDIR)
    text = (out / "summary.md").read_text()
    assert "failed to load" in text and "## Run provenance" in text
    assert "Invocations: 2" in text and "bbbbbbbb (dirty)" in text and "judge-x" in text
    assert "recommended `small`" in text and "not evaluated" in text
    assert "judged 1, judge_failed 1" in text
    assert "(n=" in text and "undefined" in text
    cells = [c.strip() for line in text.splitlines() if line.startswith("|") for c in line.split("|")]
    assert not any("nan" in c.lower().split() or c.lower().startswith("nan") for c in cells)


def test_triage_failures_count_against_baseline():
    rows = [triage_row("hosted", f"h{i}", 5) for i in range(10)]
    for i in range(10):
        if i < 2:
            rows.append(triage_row("m", f"h{i}", 5))
        else:
            r = resp("m", "triage", f"h{i}", 1, 120.0, {"failed": True})
            r["result"]["error"] = "timeout"
            rows.append(r)
    metrics = agg(rows)
    assert metrics["m"]["triage"]["quality_vs_baseline"] < -0.5
    assert metrics["m"]["triage"]["n_metrics"]["quality_vs_baseline"] == 10
    acc = {"t": {"task": "triage", "criteria": {"quality_vs_baseline": {"min": -0.1}}}}
    assert evaluate(acc, metrics)["t"]["m"]["pass"] is False


def test_lower_is_better_rates_exclude_failed_items():
    rows = tool_rows("m", 10, 1.0)
    rows.append(timeout_row("m", "x"))
    assert agg(rows)["m"]["tools"]["n_metrics"]["invalid_rate"] == 10
    rag = [rag_row("m", "a", True, 1.0, True)]
    f = resp("m", "rag", "b", 1, 1.0, {"failed": True})
    f["result"]["error"] = "timeout"
    m = agg(rag + [f])["m"]["rag"]
    assert m["false_refusal"] == 1.0 and m["n_metrics"]["false_refusal"] == 1
