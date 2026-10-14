"""Run the model grid and write raw responses. Scoring is a separate step (evalh.score)."""
import argparse
import asyncio
import datetime as dt
import os
import random
from pathlib import Path

from evalh.clients.ollama import OllamaClient
from evalh.clients.openai_compat import OpenAICompatClient
from evalh.config import ModelConfig, load_grid
from evalh.data import append_jsonl, read_jsonl
from evalh.prompts import rag_messages, tool_messages, triage_messages
from evalh.rag.index import Index
from evalh.sysinfo import MemorySampler, git_state, machine_info, ollama_env
from evalh.tools.noc import TOOLS

ROOT = Path(__file__).resolve().parents[1]
TOP_K = 4
EMBED_MODEL = "nomic-embed-text"
EXPECTED_ERRORS = ("timeout", "first_token_timeout")


def done_keys(records: list[dict]) -> set[tuple]:
    return {(r["config"], r["task"], r["item_id"], r["repeat"]) for r in records if r.get("type") == "response"}


def order_items(items: list[dict], repeat: int) -> list[dict]:
    out = list(items)
    random.Random(1000 + repeat).shuffle(out)
    return out


def is_unexpected_error(error: str | None) -> bool:
    """Timeouts and HTTP errors are normal outcomes; anything else may be a code bug caught by the client."""
    return bool(error) and error not in EXPECTED_ERRORS and not error.startswith("http ")


def hosted_client() -> OpenAICompatClient:
    return OpenAICompatClient(os.environ["HOSTED_BASE_URL"], os.environ["HOSTED_API_KEY"], os.environ["HOSTED_MODEL"])


async def retrieve_all(ollama: OllamaClient, index: Index, questions: list[dict]) -> dict[str, list[str]]:
    vecs = await ollama.embed(EMBED_MODEL, ["search_query: " + q["question"] for q in questions])
    return {q["id"]: [cid for cid, _ in index.search(v, TOP_K)] for q, v in zip(questions, vecs)}


def build_messages(task: str, item: dict, index: Index, retrieved: dict) -> list[dict]:
    if task == "rag":
        return rag_messages(item["question"], [index.get(c) for c in retrieved[item["id"]]])
    if task == "tools":
        return tool_messages(item["question"])
    return triage_messages(item["log"])


def pending(model: ModelConfig, tasks, items, done, repeats) -> int:
    return sum((model.name, t, i["id"], r) not in done
               for r in range(1, repeats + 1) for t in tasks for i in items[t])


async def run_model(model: ModelConfig, client, ollama, tasks, items, index, retrieved, defaults, out, done,
                    run_id, repeats):
    options = {k: defaults[k] for k in ("temperature", "seed", "num_ctx", "num_predict")}
    is_local = model.provider == "ollama"
    meta = {"type": "model_meta", "run_id": run_id, "config": model.name, "provider": model.provider,
            "family": model.family, "params_b": model.params_b, "quant": model.quant}

    with MemorySampler() as mem:
        if is_local:
            warm = await client.chat(model.name, [{"role": "user", "content": "ok"}],
                                     options={**options, "num_predict": 1})
            if not warm.ok:
                print(f"!! warmup failed for {model.name}: {warm.error}", flush=True)
            meta["warmup_load_s"] = warm.load_s
            loaded = next((m for m in await client.ps() if m["name"] == model.name), {})
            meta.update({k: loaded.get(k) for k in ("digest", "size", "size_vram", "context_length")})

        for repeat in range(1, repeats + 1):
            for task in tasks:
                for item in order_items(items[task], repeat):
                    key = (model.name, task, item["id"], repeat)
                    if key in done:
                        continue
                    result = await client.chat(model.name, build_messages(task, item, index, retrieved),
                                               tools=TOOLS if task == "tools" else None, options=options)
                    append_jsonl(out, {"type": "response", "run_id": run_id, "config": model.name, "task": task,
                                       "item_id": item["id"], "repeat": repeat,
                                       "retrieved": retrieved.get(item["id"]) if task == "rag" else None,
                                       "result": result.to_dict(), "ts": dt.datetime.now(dt.UTC).isoformat()})
                    status = "ok" if result.ok else result.error
                    print(f"{model.name} r{repeat} {item['id']}: {result.total_s:.1f}s {status}", flush=True)
                    if is_unexpected_error(result.error):
                        print(f"!! UNEXPECTED ERROR {model.name} {item['id']}: {result.error}", flush=True)

    meta["peak_runner_rss"] = mem.peak_runner_rss if is_local else None
    meta["system_used_delta"] = (mem.peak_system_used - mem.baseline_system_used) if is_local else None
    append_jsonl(out, meta)
    if is_local:
        await ollama.unload(model.name)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="rag,tools,triage")
    ap.add_argument("--only", nargs="*", help="model names to run (default: all)")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--run-id", default=dt.date.today().isoformat())
    args = ap.parse_args()

    grid = load_grid(ROOT / "configs/models.yaml")
    tasks = args.tasks.split(",")
    repeats = args.repeats or grid.defaults["repeats"]
    out = ROOT / "results" / f"raw-{args.run_id}.jsonl"
    existing = read_jsonl(out)
    done = done_keys(existing)

    async with OllamaClient() as ollama:
        items = {t: read_jsonl(ROOT / f"data/questions/{t}.jsonl") for t in tasks}
        index = Index.load(ROOT / "data/index")
        retrieved = await retrieve_all(ollama, index, items["rag"]) if "rag" in items else {}

        if not any(r.get("type") == "run_meta" for r in existing):
            append_jsonl(out, {"type": "run_meta", "run_id": args.run_id,
                               "started": dt.datetime.now(dt.UTC).isoformat(), "machine": machine_info(),
                               "ollama_version": await ollama.version(), "ollama_env": ollama_env(),
                               "git": git_state(), "defaults": grid.defaults})

        for model in grid.models:
            if args.only and model.name not in args.only:
                continue
            if model.provider != "ollama" and "HOSTED_API_KEY" not in os.environ:
                print(f"skipping {model.name}: HOSTED_API_KEY not set")
                continue
            if pending(model, tasks, items, done, repeats) == 0:
                print(f"skipping {model.name}: nothing left to run")
                continue
            for m in await ollama.ps():
                await ollama.unload(m["name"])
            if model.provider == "ollama":
                await run_model(model, ollama, ollama, tasks, items, index, retrieved, grid.defaults, out,
                                done, args.run_id, repeats)
            else:
                async with hosted_client() as client:
                    await run_model(model, client, ollama, tasks, items, index, retrieved, grid.defaults, out,
                                    done, args.run_id, repeats)


if __name__ == "__main__":
    asyncio.run(main())
