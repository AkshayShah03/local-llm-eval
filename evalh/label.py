"""Hand-label a sample of answers and measure how well the LLM judge agrees."""
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from evalh.data import append_jsonl, read_jsonl
from evalh.scoring.kappa import cohen_kappa

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/labels.jsonl"


def _is_score(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _judged(scored: list[dict]) -> dict[tuple[str, str], dict]:
    """Repeat-1 response records that carry an integer judge correctness score."""
    return {(r["config"], r["item_id"]): r for r in scored
            if r.get("type") == "response" and r.get("repeat") == 1
            and _is_score((r.get("scores") or {}).get("correctness"))}


def sample_for_labelling(scored: list[dict], n: int, seed: int = 7) -> list[dict]:
    by_config = defaultdict(list)
    for r in _judged(scored).values():
        by_config[r["config"]].append(r)
    rng = random.Random(seed)
    pools = {c: rng.sample(rows, len(rows)) for c, rows in sorted(by_config.items())}
    out = []
    while len(out) < n and any(pools.values()):
        for c in sorted(pools):
            if pools[c] and len(out) < n:
                out.append(pools[c].pop())
    return out


def _kappa(a, b, **kw) -> float | None:
    """Kappa, or None when undefined so the result stays JSON-safe."""
    k = cohen_kappa(a, b, **kw)
    return None if math.isnan(k) else k


def agreement(labels: list[dict], scored: list[dict]) -> dict:
    judged = _judged(scored)
    latest = {(l["config"], l["item_id"]): l["correctness"] for l in labels
              if _is_score(l.get("correctness")) and 1 <= l["correctness"] <= 5}
    pairs = [(h, judged[k]["scores"]["correctness"]) for k, h in latest.items() if k in judged]
    if not pairs:
        return {"n": 0, "kappa_binary": None, "kappa_quadratic": None, "exact_agreement": None}
    human, model = [p[0] for p in pairs], [p[1] for p in pairs]
    return {
        "n": len(pairs),
        "kappa_binary": _kappa([h >= 4 for h in human], [m >= 4 for m in model]),
        "kappa_quadratic": _kappa(human, model, labels=[1, 2, 3, 4, 5], weights="quadratic"),
        "exact_agreement": sum(h == m for h, m in pairs) / len(pairs),
    }


def format_report(result: dict) -> str:
    def fmt(v):
        return "undefined" if v is None else f"{v:.3f}"
    return "\n".join([f"n: {result['n']}",
                      f"kappa_binary: {fmt(result['kappa_binary'])}",
                      f"kappa_quadratic: {fmt(result['kappa_quadratic'])}",
                      f"exact_agreement: {fmt(result['exact_agreement'])}"])


def _show(q: dict, record: dict, print_fn=print) -> None:
    """Show the question, reference facts and answer. Config name and judge score stay hidden."""
    facts = "\n".join("  - " + (" | ".join(f) if isinstance(f, list) else f) for f in q["required_facts"])
    print_fn("\n" + "=" * 80)
    print_fn("QUESTION:", q.get("question") or q["log"][:800])
    print_fn("REFERENCE FACTS (any alternative on a line counts):\n" + facts)
    print_fn("-" * 80)
    print_fn(record["result"]["content"])
    print_fn("-" * 80)


def main(argv=None, input_fn=input, labels_path: Path = LABELS, root: Path = ROOT) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    scored = read_jsonl(root / f"results/scored-{args.run_id}.jsonl")
    if args.report:
        result = agreement(read_jsonl(labels_path), scored)
        print(format_report(result))
        print(json.dumps(result, indent=2))
        return

    items = {q["id"]: q for t in ("rag", "triage") for q in read_jsonl(root / f"data/questions/{t}.jsonl")}
    done = {(l["config"], l["item_id"]) for l in read_jsonl(labels_path)}
    for r in sample_for_labelling(scored, args.n):
        if (r["config"], r["item_id"]) in done or r["item_id"] not in items:
            continue
        _show(items[r["item_id"]], r)
        score = ""
        while score not in {"1", "2", "3", "4", "5"}:
            score = input_fn("correctness 1-5 (judge hidden): ").strip()
        append_jsonl(labels_path, {"config": r["config"], "item_id": r["item_id"],
                                   "correctness": int(score), "labeller": "human"})


if __name__ == "__main__":
    main()
