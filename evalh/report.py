"""Aggregate scored results, apply acceptance criteria and render the summary."""
import argparse
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evalh.config import load_acceptance  # noqa: E402
from evalh.data import read_jsonl  # noqa: E402
from evalh.scoring.stats import percentile  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MIN_FRESH_PREFILL_TOKENS = 64
HOSTED = "hosted"
PCT = {"fact_recall", "refusal_unanswerable", "false_refusal", "retrieval_hit", "tool_selection", "args_exact",
       "abstain_ok", "invalid_rate", "failure_rate", "quality_vs_baseline"}
N_COLUMNS = PCT | {"faithfulness", "correctness"}
TABLES = {
    "rag": ["fact_recall", "faithfulness", "correctness", "refusal_unanswerable", "false_refusal", "retrieval_hit",
            "latency_p50_s", "latency_p95_s", "failure_rate"],
    "tools": ["tool_selection", "args_exact", "abstain_ok", "invalid_rate", "latency_p50_s", "latency_p95_s",
              "failure_rate"],
    "triage": ["fact_recall", "correctness", "quality_vs_baseline", "latency_p50_s", "failure_rate"],
}


def _vals(values):
    return [float(v) for v in values if v is not None]


def _mean(values):
    vals = _vals(values)
    return mean(vals) if vals else None


def _median(values):
    vals = _vals(values)
    return median(vals) if vals else None


def _decode(rows):
    return [r["result"].get("decode_tps") for r in rows if r["result"].get("decode_tps")]


def _prefill(rows):
    """Prefill rates only from responses with enough freshly prefilled tokens; small counts are noisy."""
    out = []
    for r in rows:
        res = r["result"]
        prompt = res.get("prompt_tokens")
        if not prompt or not res.get("prefill_tps"):
            continue
        if prompt - (res.get("cached_prompt_tokens") or 0) >= MIN_FRESH_PREFILL_TOKENS:
            out.append(res["prefill_tps"])
    return out


def _task_metrics(task: str, rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["scores"].get("failed")]
    first = [r for r in ok if r["repeat"] == 1]
    s = [r["scores"] for r in first]
    n_of = {}

    def mean_of(name, values):
        vals = _vals(values)
        n_of[name] = len(vals)
        return mean(vals) if vals else None

    decode, prefill = _decode(ok), _prefill(ok)
    n_of["decode_tps_median"], n_of["prefill_tps_median"] = len(decode), len(prefill)
    m = {
        "n": len(rows),
        "failure_rate": 1 - len(ok) / len(rows),
        "latency_p50_s": percentile([r["result"]["total_s"] for r in ok], 50),
        "latency_p95_s": percentile([r["result"]["total_s"] for r in ok], 95),
        "ttft_p50_s": percentile([r["result"].get("ttft_s") for r in ok], 50),
        "decode_tps_median": _median(decode),
        "prefill_tps_median": _median(prefill),
    }
    if task == "rag":
        ans = [x for x in s if x["answerable"]]
        unans = [x for x in s if not x["answerable"]]
        m.update({
            "fact_recall": mean_of("fact_recall", (x.get("fact_recall") for x in ans)),
            "faithfulness": mean_of("faithfulness", (x.get("faithfulness") for x in ans if not x.get("judge_failed"))),
            "correctness": mean_of("correctness", (x.get("correctness") for x in ans if not x.get("judge_failed"))),
            "refusal_unanswerable": mean_of("refusal_unanswerable", (x.get("refused") for x in unans)),
            "false_refusal": mean_of("false_refusal", (x.get("refused") for x in ans)),
            "retrieval_hit": mean_of("retrieval_hit", (x.get("retrieval_hit") for x in ans)),
        })
        judged_pool = ans
    elif task == "tools":
        tool = [x for x in s if x["expects_tool"]]
        m.update({
            "tool_selection": mean_of("tool_selection", (x["selection_ok"] for x in tool)),
            "args_exact": mean_of("args_exact", (x["args_ok"] for x in tool)),
            "invalid_rate": mean_of("invalid_rate", (x["invalid"] for x in s)),
            "abstain_ok": mean_of("abstain_ok", (x["selection_ok"] for x in s if not x["expects_tool"])),
        })
        judged_pool = []
    else:
        m.update({"fact_recall": mean_of("fact_recall", (x.get("fact_recall") for x in s)),
                  "correctness": mean_of("correctness", (x.get("correctness") for x in s if not x.get("judge_failed")))})
        judged_pool = s
    if task != "tools":
        failed = sum(bool(x.get("judge_failed")) for x in judged_pool)
        m["judge_failed"] = failed
        m["judged"] = sum(("correctness" in x or "faithfulness" in x) and not x.get("judge_failed")
                          for x in judged_pool)
    m["n_metrics"] = n_of
    return m


def aggregate(scored: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in scored:
        if r.get("type") == "response" and r.get("scores") is not None:
            groups[(r["config"], r["task"])].append(r)
    out = defaultdict(dict)
    for (config, task), rows in groups.items():
        out[config][task] = _task_metrics(task, rows)
    base = out.get(HOSTED, {}).get("triage", {}).get("correctness")
    for config, tasks in out.items():
        if "triage" in tasks:
            c = tasks["triage"].get("correctness")
            tasks["triage"]["quality_vs_baseline"] = (c - base) / base if base and c is not None else None
            tasks["triage"]["n_metrics"]["quality_vs_baseline"] = tasks["triage"]["n_metrics"].get("correctness", 0)
    return dict(out)


def throughput_series(scored: list[dict]) -> dict:
    """Decode and prefill medians per config, pooled over all tasks and repeats."""
    groups = defaultdict(list)
    for r in scored:
        if r.get("type") == "response" and r.get("scores") is not None and not r["scores"].get("failed"):
            groups[r["config"]].append(r)
    return {c: {"decode_tps_median": _median(_decode(rows)), "n_decode": len(_decode(rows)),
                "prefill_tps_median": _median(_prefill(rows)), "n_prefill": len(_prefill(rows))}
            for c, rows in groups.items()}


def evaluate(acceptance: dict, metrics: dict) -> dict:
    result = {}
    for use_case, spec in acceptance.items():
        result[use_case] = {}
        for config, tasks in metrics.items():
            m = tasks.get(spec["task"])
            if m is None:
                continue
            failed, unknown = [], False
            for name, bound in spec["criteria"].items():
                value = m.get(name)
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    unknown = True
                elif ("min" in bound and value < bound["min"]) or ("max" in bound and value > bound["max"]):
                    failed.append(name)
            result[use_case][config] = {"pass": False if failed else (None if unknown else True), "failed": failed}
    return result


def recommend(evaluation: dict, memory: dict) -> dict:
    out = {}
    for use_case, configs in evaluation.items():
        local = [c for c, v in configs.items() if v["pass"] and c != HOSTED and memory.get(c)]
        if local:
            out[use_case] = min(local, key=lambda c: memory[c])
        else:
            out[use_case] = HOSTED if configs.get(HOSTED, {}).get("pass") else "none"
    return out


def latest_by_config(records: list[dict], rtype: str, prefer_complete: bool = False) -> dict:
    """Latest record of a type per config. With prefer_complete, the latest non-partial one wins when present."""
    out = {}
    for r in records:
        if r.get("type") != rtype:
            continue
        cur = out.get(r["config"])
        if prefer_complete and cur is not None and not cur.get("partial") and r.get("partial"):
            continue
        out[r["config"]] = r
    return out


def memory_series(order: list[str], loads: dict) -> list[tuple[str, float]]:
    """(config, GB resident in GPU memory) for local configs that loaded."""
    return [(c, loads[c]["size_vram"] / 1e9) for c in order
            if c in loads and not loads[c].get("error") and loads[c].get("size_vram")]


def fmt(v, pct=False) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float) and math.isnan(v):
        return "undefined"
    return f"{v:.0%}" if pct else (f"{v:.2f}" if isinstance(v, float) else str(v))


def _gb(b):
    return f"{b / 1e9:.1f}" if b else "–"


def _short(name: str) -> str:
    return name.replace("-instruct", "")


def _provenance(run_meta, invocations, responses, loads, metrics) -> list[str]:
    lines = ["## Run provenance", ""]
    machine = run_meta.get("machine") or {}
    lines.append(f"- Machine: {machine.get('chip', 'unknown')}, {machine.get('memory_gb', '?')} GB, "
                 f"{machine.get('os', 'unknown')}")
    lines.append(f"- Ollama {run_meta.get('ollama_version', 'unknown')}, env {run_meta.get('ollama_env', {})}")
    lines.append(f"- Invocations: {len(invocations)}")
    seen = []
    for g in [run_meta.get("git")] + [i.get("git") for i in invocations]:
        if g and g.get("sha"):
            label = f"{g['sha'][:8]}{' (dirty)' if g.get('dirty') else ''}"
            if label not in seen:
                seen.append(label)
    lines.append(f"- Git: {', '.join(seen) or 'unknown'}")
    judges = sorted({r["judge"] for r in responses if r.get("judge")})
    lines.append(f"- Judge model: {', '.join(judges) or 'none (no judge scores)'}")
    hosted = sorted({m["hosted_model"] for m in responses if m.get("hosted_model")})
    if hosted:
        lines.append(f"- Hosted baseline model: {', '.join(hosted)}")
    failed_load = [c for c, ld in loads.items() if ld.get("error")]
    lines.append(f"- Failed to load: {', '.join(failed_load) or 'none'}")
    bad = [f"{c}/{t} ({m['failure_rate']:.0%} of {m['n']})" for c, ts in metrics.items()
           for t, m in ts.items() if m["failure_rate"] > 0]
    lines.append(f"- Configs with failed responses: {', '.join(bad) or 'none'}")
    judge_rows = [f"{c}/{t}: judged {m['judged']}, judge_failed {m['judge_failed']}"
                  for c, ts in metrics.items() for t, m in ts.items() if "judged" in m]
    if judge_rows:
        lines += ["- Judge coverage (repeat 1):"] + [f"  - {j}" for j in judge_rows]
    return lines


def _header(task, col, order, metrics):
    if col not in N_COLUMNS:
        return col
    ns = [metrics[c][task]["n_metrics"].get(col) for c in order if task in metrics.get(c, {})]
    ns = [n for n in ns if n]
    return f"{col} (n={max(ns)})" if ns else col


def _cell(task, col, t, header_n):
    v = t.get(col)
    s = fmt(v, col in PCT)
    n = t.get("n_metrics", {}).get(col)
    if v is not None and col in N_COLUMNS and n and header_n is not None and n != header_n:
        s += f" (n={n})"
    return s


def render(run_id: str, results_dir: Path | None = None, out_dir: Path | None = None,
           acceptance_path: Path | None = None, charts: bool = True) -> Path:
    results_dir = Path(results_dir or ROOT / "results")
    out = Path(out_dir or ROOT / "reports")
    scored = read_jsonl(results_dir / f"scored-{run_id}.jsonl")
    run_meta = next((r for r in scored if r.get("type") == "run_meta"), {})
    invocations = [r for r in scored if r.get("type") == "invocation"]
    loads = latest_by_config(scored, "model_load")
    meta = latest_by_config(scored, "model_meta", prefer_complete=True)
    metrics = aggregate(scored)
    acceptance = load_acceptance(Path(acceptance_path or ROOT / "configs/acceptance.yaml"))
    ev = evaluate(acceptance, metrics)
    memory = {c: (None if ld.get("error") else ld.get("size_vram")) for c, ld in loads.items()}
    rec = recommend(ev, memory)
    order = list(dict.fromkeys(r["config"] for r in scored if r.get("config") and r["type"] != "invocation"))
    tput = throughput_series(scored)

    lines = [f"# Results: run {run_id}", ""]
    lines += _provenance(run_meta, invocations,
                         [r for r in scored if r.get("type") == "response"] + list(meta.values()), loads, metrics)

    lines += ["", "## Memory and throughput", "",
              "| config | weights+KV in GPU (GB) | peak runner RSS (GB) | system delta (GB) | load s | "
              "decode tok/s median (n) | prefill tok/s median (n, >= 64 fresh tokens) |",
              "|---|---|---|---|---|---|---|"]
    for c in order:
        ld, mt, tp = loads.get(c, {}), meta.get(c, {}), tput.get(c, {})
        if ld.get("error"):
            lines.append(f"| {c} | failed to load | – | – | – | – | – |")
            continue
        dec = f"{fmt(tp.get('decode_tps_median'))} ({tp.get('n_decode', 0)})" if tp else "–"
        pre = f"{fmt(tp.get('prefill_tps_median'))} ({tp.get('n_prefill', 0)})" if tp else "–"
        lines.append(f"| {c} | {_gb(ld.get('size_vram'))} | {_gb(mt.get('peak_runner_rss'))} | "
                     f"{_gb(mt.get('system_used_delta'))} | {fmt(ld.get('warmup_load_s'))} | {dec} | {pre} |")

    for task, cols in TABLES.items():
        heads = [_header(task, k, order, metrics) for k in cols]
        header_n = {k: max([metrics[c][task]["n_metrics"].get(k) or 0 for c in order if task in metrics.get(c, {})]
                           or [0]) or None for k in cols if k in N_COLUMNS}
        lines += ["", f"## {task}", "", "| config | " + " | ".join(heads) + " |", "|---" * (len(cols) + 1) + "|"]
        for c in order:
            if loads.get(c, {}).get("error"):
                lines.append(f"| {c} | failed to load |" + " – |" * (len(cols) - 1))
                continue
            t = metrics.get(c, {}).get(task, {})
            lines.append(f"| {c} | " + " | ".join(_cell(task, k, t, header_n.get(k)) for k in cols) + " |")

    lines += ["", "## Acceptance", ""]
    for use_case, configs in ev.items():
        r = rec[use_case]
        if r == "none" and any(v["pass"] is None for v in configs.values()):
            r = "none (some configs not evaluated)"
        lines += [f"### {use_case}: recommended `{r}`", "", "| config | pass | failed criteria |", "|---|---|---|"]
        for c in order:
            if c in configs:
                v = configs[c]
                lines.append(f"| {c} | {fmt(v['pass']) if v['pass'] is not None else 'not evaluated'} | "
                             f"{', '.join(v['failed']) or '–'} |")
            elif loads.get(c, {}).get("error"):
                lines.append(f"| {c} | failed to load | – |")
        lines.append("")

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    if charts:
        _charts(order, metrics, loads, tput, out)
    print(f"wrote {out / 'summary.md'}")
    return out / "summary.md"


def _charts(order, metrics, loads, tput, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (task, metric, label) in zip(axes, [("tools", "args_exact", "Tool calls: exact-argument accuracy"),
                                               ("rag", "fact_recall", "Document Q&A: fact recall")]):
        for c in order:
            t = metrics.get(c, {}).get(task, {})
            if t.get(metric) is not None and t.get("latency_p95_s") is not None:
                ax.scatter(t["latency_p95_s"], t[metric], color="#2a6f97")
                ax.annotate(_short(c), (t["latency_p95_s"], t[metric]), fontsize=8,
                            xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("p95 latency (seconds)")
        ax.set_ylabel(f"{metric} (fraction)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(label)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "quality_vs_latency.png", dpi=150)
    plt.close(fig)

    mem = memory_series(order, loads)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([_short(c) for c, _ in mem], [g for _, g in mem], color="#2a6f97")
    ax.axvline(16, color="#b23a48", linestyle="--", linewidth=1, label="16 GB machine memory")
    ax.set_xlabel("GB resident in GPU memory (weights + KV cache, 8k context)")
    ax.legend()
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out / "memory.png", dpi=150)
    plt.close(fig)

    names = [c for c, _ in mem]
    fig, ax = plt.subplots(figsize=(9, 5))
    ys = range(len(names))
    dec = [tput.get(c, {}).get("decode_tps_median") or 0 for c in names]
    pre = [tput.get(c, {}).get("prefill_tps_median") or 0 for c in names]
    ax.barh([y + 0.2 for y in ys], dec, height=0.4, label="decode", color="#2a6f97")
    ax.barh([y - 0.2 for y in ys], pre, height=0.4, label="prefill (>= 64 fresh tokens)", color="#e09f3e")
    ax.set_yticks(list(ys), [_short(c) for c in names], fontsize=8)
    ax.set_xlabel("tokens per second (median)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "throughput.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    render(ap.parse_args().run_id)
