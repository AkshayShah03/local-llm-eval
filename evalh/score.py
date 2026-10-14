"""Add deterministic and judge scores to raw responses."""
import argparse
import asyncio
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from evalh.clients.ollama import OllamaClient
from evalh.data import append_jsonl, read_jsonl
from evalh.rag.index import Index
from evalh.run import hosted_client
from evalh.scoring.facts import fact_recall, is_refusal
from evalh.scoring.judge import judge
from evalh.scoring.retrieval import hit_at_k
from evalh.scoring.toolcalls import score_tool_calls

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("rag", "tools", "triage")


def deterministic_scores(task: str, item: dict, record: dict) -> dict:
    res = record["result"]
    if res.get("error"):
        return {"failed": True}
    if task == "rag":
        gold = item["gold_chunk_ids"]
        return {"failed": False,
                "fact_recall": fact_recall(res["content"], item["required_facts"])[0] if item["answerable"] else None,
                "refused": is_refusal(res["content"]), "answerable": item["answerable"],
                "retrieval_hit": hit_at_k(record["retrieved"], gold, k=4) if gold else None}
    if task == "tools":
        ts = score_tool_calls(res["tool_calls"], item["expected_calls"], res["content"])
        return {"failed": False, "selection_ok": ts.selection_ok, "args_ok": ts.args_ok,
                "invalid": ts.invalid, "expects_tool": bool(item["expected_calls"])}
    return {"failed": False, "fact_recall": fact_recall(res["content"], item["required_facts"])[0]}


def judge_inputs(task: str, item: dict, record: dict, index: Index) -> tuple[str, list, str] | None:
    if task == "rag" and item["answerable"]:
        context = "\n\n".join(index.get(c)["text"] for c in record["retrieved"])
        return item["question"], item["required_facts"], context
    if task == "triage":
        return "Summarise this incident for the on-call engineer.", item["required_facts"], item["log"]
    return None


async def score_run(raw: list[dict], items: dict, index: Index, judge_client, judge_name: str,
                    cache_path: Path, out: Path) -> None:
    """Write scored records to out. Successful judgements are cached; failures are not, so a rerun retries them."""
    cache = {(c["config"], c["item_id"], c["judge"]): c["scores"] for c in read_jsonl(cache_path)}
    out.unlink(missing_ok=True)
    for r in raw:
        if r["type"] != "response":
            append_jsonl(out, r)
            continue
        task, item = items[r["item_id"]]
        scores = deterministic_scores(task, item, r)
        inputs = judge_inputs(task, item, r, index)
        if inputs and r["repeat"] == 1 and not scores["failed"]:
            key = (r["config"], r["item_id"], judge_name)
            if key not in cache:
                verdict = await judge(judge_client, judge_name, *inputs, r["result"]["content"])
                if verdict is None:
                    print(f"warning: judge failed for {r['config']} {r['item_id']}", file=sys.stderr, flush=True)
                    scores["judge_failed"] = True
                else:
                    cache[key] = verdict
                    append_jsonl(cache_path, {"config": key[0], "item_id": key[1], "judge": key[2],
                                              "scores": verdict})
            if key in cache:
                scores.update({"correctness": cache[key]["correctness"],
                               "faithfulness": cache[key]["faithfulness"], "judge_reason": cache[key]["reason"]})
        append_jsonl(out, {**r, "scores": scores, "judge": judge_name})


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--judge", default="hosted", help="'hosted' or an Ollama model name")
    args = ap.parse_args()
    if args.judge == "hosted":
        missing = [v for v in ("HOSTED_BASE_URL", "HOSTED_API_KEY", "HOSTED_MODEL") if v not in os.environ]
        if missing:
            print(f"error: --judge hosted requires {', '.join(missing)} to be set", file=sys.stderr)
            sys.exit(2)

    raw = read_jsonl(ROOT / f"results/raw-{args.run_id}.jsonl")
    items = {q["id"]: (t, q) for t in TASKS for q in read_jsonl(ROOT / f"data/questions/{t}.jsonl")}
    index = Index.load(ROOT / "data/index")
    out = ROOT / f"results/scored-{args.run_id}.jsonl"
    async with AsyncExitStack() as stack:
        client = hosted_client() if args.judge == "hosted" else OllamaClient()
        judge_client = await stack.enter_async_context(client)
        judge_name = args.judge
        await score_run(raw, items, index, judge_client, judge_name, ROOT / "results/judge-cache.jsonl", out)
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
