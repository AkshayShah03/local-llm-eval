"""Run the model grid and write raw responses. Scoring is a separate step (evalh.score).

Raw file records (results/raw-<run_id>.jsonl):
  run_meta    first record, written once.
  invocation  one per process start: git state, CLI args, effective settings, input hashes.
  model_load  right after warmup: digest, size, size_vram, context_length, warmup_load_s,
              system_used_delta_after_load.
  response    one per (config, task, item_id, repeat); only real outcomes (success, timeout,
              first_token_timeout, http 4xx). Anything else is an outage or a bug: it goes to
              results/errors-<run_id>.jsonl and is NOT counted as done, so a resume redoes it.
  model_meta  end of a model's window: peaks, responses_in_window, partial.

Reports should use the latest model_load per config, and the non-partial model_meta when one
exists (a resumed model's peaks only cover the resumed window, marked partial: true).
"""
import argparse
import asyncio
import datetime as dt
import hashlib
import os
import random
import sys
from pathlib import Path

import httpx

from evalh.clients.ollama import OllamaClient
from evalh.clients.openai_compat import OpenAICompatClient
from evalh.config import ModelConfig, load_grid
from evalh.data import append_jsonl, read_jsonl
from evalh.prompts import rag_messages, tool_messages, triage_messages
from evalh.rag.index import Index
from evalh.sysinfo import MemorySampler, git_state, machine_info, ollama_env, system_used
from evalh.tools.noc import TOOLS

ROOT = Path(__file__).resolve().parents[1]
TOP_K = 4
EMBED_MODEL = "nomic-embed-text"
# Connection-level failures from httpx, formatted by the client as "<ExcName>: ...".
TRANSPORT_ERRORS = ("ConnectError", "ReadError", "WriteError", "CloseError", "RemoteProtocolError",
                    "LocalProtocolError", "NetworkError", "ProxyError", "UnsupportedProtocol")
OUTAGE_HTTP = ("http 502", "http 503", "http 504")
MAX_CONSECUTIVE_ERRORS = 5
HOSTED_VARS = ("HOSTED_BASE_URL", "HOSTED_API_KEY", "HOSTED_MODEL")


class RunAborted(Exception):
    pass


class WarmupFailed(Exception):
    """The model could not be loaded (for example out of memory); the grid moves on to the next model."""


def done_keys(records: list[dict]) -> set[tuple]:
    return {(r["config"], r["task"], r["item_id"], r["repeat"]) for r in records if r.get("type") == "response"}


def order_items(items: list[dict], repeat: int) -> list[dict]:
    out = list(items)
    random.Random(1000 + repeat).shuffle(out)
    if repeat > 1 and len(out) > 1:
        # Deterministic from the item list alone, so a resume reproduces the same order.
        if out[0]["id"] == order_items(items, repeat - 1)[-1]["id"]:
            out = out[1:] + out[:1]
    return out


def is_outage(error: str | None) -> bool:
    """Only transport failures and gateway-style 502/503/504 mean the server is unavailable. Those are
    retried on resume. Every other error (in-stream errors, other 5xx, decode errors) can be triggered
    deterministically by the prompt, so it is recorded as a failed response and scored."""
    if not error:
        return False
    if error.startswith("http "):
        return error[:8] in OUTAGE_HTTP
    return error.split(":", 1)[0] in TRANSPORT_ERRORS


def undone_keys(models: list[ModelConfig], tasks, items, repeats, done) -> list[tuple]:
    return [(m.name, t, i["id"], r) for m in models for r in range(1, repeats + 1)
            for t in tasks for i in items[t] if (m.name, t, i["id"], r) not in done]


def sha256_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def input_hashes() -> dict:
    files = sorted((ROOT / "data/questions").glob("*.jsonl")) + [ROOT / "data/index/chunks.jsonl",
                                                                ROOT / "data/index/embeddings.npy"]
    return {str(f.relative_to(ROOT)): sha256_file(f) for f in files}


def hosted_env() -> dict | None:
    return {k: os.environ[k] for k in HOSTED_VARS} if all(os.environ.get(k) for k in HOSTED_VARS) else None


def hosted_client(env: dict) -> OpenAICompatClient:
    return OpenAICompatClient(env["HOSTED_BASE_URL"], env["HOSTED_API_KEY"], env["HOSTED_MODEL"])


async def retrieve_all(ollama: OllamaClient, index: Index, questions: list[dict]) -> dict[str, list[str]]:
    vecs = await ollama.embed(EMBED_MODEL, ["search_query: " + q["question"] for q in questions])
    return {q["id"]: [cid for cid, _ in index.search(v, TOP_K)] for q, v in zip(questions, vecs)}


async def embed_digest(ollama: OllamaClient) -> str | None:
    for m in await ollama.tags():
        if m.get("name") in (EMBED_MODEL, f"{EMBED_MODEL}:latest"):
            return m.get("digest")
    return None


def build_messages(task: str, item: dict, index: Index, retrieved: dict) -> list[dict]:
    if task == "rag":
        return rag_messages(item["question"], [index.get(c) for c in retrieved[item["id"]]])
    if task == "tools":
        return tool_messages(item["question"])
    return triage_messages(item["log"])


def pending(model: ModelConfig, tasks, items, done, repeats) -> int:
    return sum((model.name, t, i["id"], r) not in done
               for r in range(1, repeats + 1) for t in tasks for i in items[t])


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


async def run_model(model: ModelConfig, client, ollama, tasks, items, index, retrieved, defaults, out, done,
                    run_id, repeats, errors_path=None, hosted=None, max_consecutive=MAX_CONSECUTIVE_ERRORS):
    options = {k: defaults[k] for k in ("temperature", "seed", "num_ctx", "num_predict")}
    is_local = model.provider == "ollama"
    errors_path = errors_path or Path(out).with_name(f"errors-{run_id}.jsonl")
    partial = any(k[0] == model.name for k in done)
    base = {"run_id": run_id, "config": model.name}
    written = 0
    consecutive = 0

    with MemorySampler() as mem:
        if is_local:
            warm = await client.chat(model.name, [{"role": "user", "content": "ok"}],
                                     options={**options, "num_predict": 1})
            if not warm.ok:
                if is_outage(warm.error):
                    raise RunAborted(f"warmup of {model.name} hit an outage ({warm.error}); rerun to resume.")
                print(f"!! warmup failed for {model.name}: {warm.error}; skipping this model",
                      file=sys.stderr, flush=True)
                append_jsonl(out, {"type": "model_load", **base, "provider": model.provider,
                                   "error": warm.error, "ts": _now()})
                await ollama.unload(model.name)
                raise WarmupFailed(f"{model.name}: {warm.error}")
            loaded = next((m for m in await client.ps() if m["name"] == model.name), {})
            append_jsonl(out, {"type": "model_load", **base, "provider": model.provider,
                               "digest": loaded.get("digest"), "size": loaded.get("size"),
                               "size_vram": loaded.get("size_vram"), "context_length": loaded.get("context_length"),
                               "warmup_load_s": warm.load_s,
                               "system_used_delta_after_load": system_used() - mem.baseline_system_used,
                               "ts": _now()})

        for repeat in range(1, repeats + 1):
            for task in tasks:
                for item in order_items(items[task], repeat):
                    key = (model.name, task, item["id"], repeat)
                    if key in done:
                        continue
                    result = await client.chat(model.name, build_messages(task, item, index, retrieved),
                                               tools=TOOLS if task == "tools" else None, options=options)
                    if is_outage(result.error):
                        consecutive += 1
                        msg = f"!! OUTAGE {model.name} {item['id']}: {result.error}"
                        print(msg, file=sys.stderr, flush=True)
                        append_jsonl(errors_path, {**base, "task": task, "item_id": item["id"],
                                                   "repeat": repeat, "result": result.to_dict(), "ts": _now()})
                        if consecutive >= max_consecutive:
                            raise RunAborted(
                                f"{consecutive} consecutive unexpected errors on {model.name} (last: "
                                f"{result.error}); aborting. Fix the cause and rerun the same command to resume; "
                                f"failed items were not recorded as done.")
                        continue
                    consecutive = 0
                    if result.error and result.error not in ("timeout", "first_token_timeout") \
                            and not result.error.startswith("http 4"):
                        print(f"!! recorded failed response {model.name} {item['id']}: {result.error}",
                              file=sys.stderr, flush=True)
                    append_jsonl(out, {"type": "response", **base, "task": task,
                                       "item_id": item["id"], "repeat": repeat,
                                       "retrieved": retrieved.get(item["id"]) if task == "rag" else None,
                                       "result": result.to_dict(), "ts": _now()})
                    written += 1
                    print(f"{model.name} r{repeat} {item['id']}: {result.total_s:.1f}s "
                          f"{'ok' if result.ok else result.error}", flush=True)

    meta = {"type": "model_meta", **base, "provider": model.provider, "family": model.family,
            "params_b": model.params_b, "quant": model.quant, "responses_in_window": written, "partial": partial,
            "peak_runner_rss": mem.peak_runner_rss if is_local else None,
            "system_used_delta": (mem.peak_system_used - mem.baseline_system_used) if is_local else None}
    if hosted:
        meta.update({"hosted_model": hosted["HOSTED_MODEL"], "hosted_base_url": hosted["HOSTED_BASE_URL"]})
    append_jsonl(out, meta)
    if is_local:
        await ollama.unload(model.name)


async def _run_one(model, hosted, ollama, tasks, items, index, retrieved, defaults, out, done, run_id, repeats):
    if pending(model, tasks, items, done, repeats) == 0:
        print(f"skipping {model.name}: nothing left to run")
        return
    for m in await ollama.ps():
        await ollama.unload(m["name"])
    if model.provider == "ollama":
        await run_model(model, ollama, ollama, tasks, items, index, retrieved, defaults, out, done, run_id,
                        repeats)
    else:
        async with hosted_client(hosted) as client:
            await run_model(model, client, ollama, tasks, items, index, retrieved, defaults, out, done, run_id,
                            repeats, hosted=hosted)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="rag,tools,triage")
    ap.add_argument("--only", nargs="*", help="model names to run (default: all)")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--run-id", default=dt.date.today().isoformat())
    ap.add_argument("--allow-changed-inputs", action="store_true",
                    help="continue a run even if question or index files changed since its first invocation")
    args = ap.parse_args()

    grid = load_grid(ROOT / "configs/models.yaml")
    tasks = args.tasks.split(",")
    repeats = args.repeats or grid.defaults["repeats"]
    out = ROOT / "results" / f"raw-{args.run_id}.jsonl"
    existing = read_jsonl(out)
    done = done_keys(existing)
    print(f"output: {out}; already done: {len(done)} responses", flush=True)

    hashes = input_hashes()
    prior = [r for r in existing if r.get("type") == "invocation"]
    if prior and prior[0].get("input_hashes") != hashes:
        changed = sorted(k for k in hashes if (prior[0].get("input_hashes") or {}).get(k) != hashes[k])
        msg = f"inputs changed since the first invocation: {', '.join(changed)}"
        if not args.allow_changed_inputs:
            print(f"refusing to continue: {msg}. Pass --allow-changed-inputs to override.", file=sys.stderr)
            return 1
        print(f"warning: {msg} (continuing: --allow-changed-inputs)", file=sys.stderr)

    hosted = hosted_env()
    try:
        async with OllamaClient() as ollama:
            version = await ollama.version()
            items = {t: read_jsonl(ROOT / f"data/questions/{t}.jsonl") for t in tasks}
            index = Index.load(ROOT / "data/index")
            retrieved = await retrieve_all(ollama, index, items["rag"]) if "rag" in items else {}
            digest = await embed_digest(ollama)

            if not any(r.get("type") == "run_meta" for r in existing):
                append_jsonl(out, {"type": "run_meta", "run_id": args.run_id, "started": _now(),
                                   "machine": machine_info(), "ollama_version": version,
                                   "ollama_env": ollama_env(), "git": git_state(ROOT), "defaults": grid.defaults})
            append_jsonl(out, {"type": "invocation", "run_id": args.run_id, "ts": _now(),
                               "git": git_state(ROOT), "ollama_version": version, "args": vars(args),
                               "tasks": tasks, "only": args.only, "repeats": repeats, "top_k": TOP_K,
                               "embed_model": EMBED_MODEL, "embed_model_digest": digest,
                               "input_hashes": hashes})

            requested = []
            for model in grid.models:
                if args.only and model.name not in args.only:
                    continue
                if model.provider != "ollama" and hosted is None:
                    print(f"skipping {model.name}: set all of {', '.join(HOSTED_VARS)}")
                    continue
                requested.append(model)
                try:
                    await _run_one(model, hosted, ollama, tasks, items, index, retrieved, grid.defaults, out,
                                   done, args.run_id, repeats)
                except WarmupFailed as e:
                    print(f"skipping model, could not load: {e}", file=sys.stderr, flush=True)

            left = undone_keys(requested, tasks, items, repeats, done_keys(read_jsonl(out)))
            if left:
                counts: dict = {}
                for c, t, _, r in left:
                    counts[(c, t, r)] = counts.get((c, t, r), 0) + 1
                print(f"INCOMPLETE: {len(left)} expected responses are missing:", file=sys.stderr)
                for (c, t, r), n in sorted(counts.items()):
                    print(f"  {c} {t} repeat {r}: {n}", file=sys.stderr)
                return 1
    except RunAborted as e:
        print(f"ABORTED: {e}", file=sys.stderr)
        return 1
    except httpx.HTTPError as e:
        print(f"ABORTED: Ollama request failed ({type(e).__name__}: {e}). Is the server running? "
              f"Rerun the same command to resume.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
