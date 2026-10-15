import math

import yaml

from evalh.report import (aggregate, evaluate, fmt, latest_by_config, memory_series, recommend, render,
                          throughput_series)


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
    m = aggregate(tool_rows("small", 9, 1.0))["small"]["tools"]
    assert m["tool_selection"] == 0.9 and m["n"] == 20 and m["latency_p95_s"] == 1.0
    assert m["n_metrics"]["tool_selection"] == 10


def test_evaluate_and_recommend():
    metrics = aggregate(tool_rows("small", 9, 1.0) + tool_rows("big", 10, 2.0))
    acc = {"noc_tools": {"task": "tools", "criteria": {"tool_selection": {"min": 0.95}, "latency_p95_s": {"max": 4.0}}}}
    ev = evaluate(acc, metrics)
    assert ev["noc_tools"]["small"] == {"pass": False, "failed": ["tool_selection"]}
    assert ev["noc_tools"]["big"]["pass"] is True
    assert recommend(ev, {"small": 1, "big": 5}) == {"noc_tools": "big"}


def test_missing_metric_is_not_evaluated():
    metrics = aggregate(tool_rows("small", 9, 1.0))
    acc = {"x": {"task": "tools", "criteria": {"faithfulness": {"min": 4}}}}
    assert evaluate(acc, metrics)["x"]["small"]["pass"] is None


def test_recommend_without_hosted_and_with_hosted():
    ev = {"u": {"a": {"pass": False, "failed": ["x"]}}}
    assert recommend(ev, {"a": 1}) == {"u": "none"}
    ev = {"u": {"a": {"pass": False, "failed": ["x"]}, "hosted": {"pass": True, "failed": []}}}
    assert recommend(ev, {"a": 1}) == {"u": "hosted"}


def test_failed_responses_count_in_failure_rate_only():
    rows = tool_rows("m", 10, 1.0)
    rows.append(resp("m", "tools", "t0", 3, 9.0, {"failed": True}))
    m = aggregate(rows)["m"]["tools"]
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
    m = aggregate(rows)["m"]["rag"]
    assert m["faithfulness"] == 4 and m["n_metrics"]["faithfulness"] == 1
    assert m["correctness"] == 4 and m["n_metrics"]["correctness"] == 2
    assert m["judged"] == 2 and m["judge_failed"] == 1
    assert m["fact_recall"] == 0.5 and m["refusal_unanswerable"] == 1.0 and m["false_refusal"] == 0.0


def triage_row(config, item, correctness):
    return resp(config, "triage", item, 1, 1.0, {"failed": False, "fact_recall": 1.0, "correctness": correctness})


def test_quality_vs_baseline_with_and_without_hosted():
    both = aggregate([triage_row("m", "a", 4), triage_row("hosted", "a", 5)])
    assert abs(both["m"]["triage"]["quality_vs_baseline"] - (-0.2)) < 1e-9
    local = aggregate([triage_row("m", "a", 4)])
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
    m = aggregate(rows)["m"]
    assert m["rag"]["prefill_tps_median"] == 200.0 and m["rag"]["n_metrics"]["prefill_tps_median"] == 2
    assert throughput_series(rows)["m"]["prefill_tps_median"] == 300.0


def test_latest_by_config_and_memory_series():
    recs = [{"type": "model_load", "config": "a", "size_vram": 1}, {"type": "model_load", "config": "a", "size_vram": 2},
            {"type": "model_load", "config": "b", "error": "boom"}]
    loads = latest_by_config(recs, "model_load")
    assert loads["a"]["size_vram"] == 2
    assert memory_series(["a", "b", "hosted"], loads) == [("a", 2 / 1e9 * 1.0)]


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
    ]
    results = tmp_path / "results"
    results.mkdir()
    (results / "scored-r.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in scored) + "\n")
    acc = tmp_path / "acc.yaml"
    acc.write_text(yaml.safe_dump({"use_cases": {
        "noc_tools": {"task": "tools", "criteria": {"tool_selection": {"min": 0.9}}},
        "triage": {"task": "triage", "criteria": {"quality_vs_baseline": {"min": -0.1}}}}}))
    out = tmp_path / "reports"
    render("r", results_dir=results, out_dir=out, acceptance_path=acc, charts=False)
    text = (out / "summary.md").read_text()
    assert "failed to load" in text and "## Run provenance" in text
    assert "Invocations: 2" in text and "bbbbbbbb (dirty)" in text and "judge-x" in text
    assert "recommended `small`" in text and "not evaluated" in text
    assert "judged 1, judge_failed 1" in text
    assert "(n=" in text and "nan" not in text.replace("Acceptance", "").split()
