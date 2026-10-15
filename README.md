# local-llm-eval

An evaluation harness for choosing which locally hosted open model (Qwen, Llama, Mistral on Ollama) should serve a given network-operations use case, at which quantization, and on what hardware, compared with a hosted API. It also includes a small FastAPI gateway that serves the chosen model with fallback.

## The problem

Teams that want to run open models on their own hardware usually pick one by reputation or by a public leaderboard, then find out in production that it calls the wrong tool, invents answers that are not in their documents, or does not fit in the GPU memory they have. Leaderboards measure general ability on generic data. They say nothing about whether a 7B model at 4-bit quantization can answer questions about your BGP community policy from your own documentation, emit a valid tool call against your NOC tooling, or stay under a latency budget on a 16 GB machine. Without a repeatable evaluation on your own tasks, with acceptance criteria written down before the results are in, model selection is guesswork and nobody can say when to re-run it.

## How this project solves it

The harness runs every model configuration through the same three task sets, built for an IP backbone (AS2914) setting, scores the answers, and checks them against pre-registered acceptance criteria.

- **Test set** (`data/`): 48 RAG questions over six IETF RFCs (BGP-4, communities, BLACKHOLE, large communities, RPKI origin validation, RFC 8212) and NTT's public AS2914 customer community page, including 8 questions the documents cannot answer; 25 tool-calling questions against a mock NOC toolset (BGP session state, prefix origin and RPKI state, interface errors, tickets) backed by fixture data; 10 synthetic incident logs to summarise. Only documentation IP ranges and ASNs are used in fixtures and logs.
- **Retrieval** (`evalh/rag/`): RFC page furniture and boilerplate are stripped, documents are chunked without breaking tables or separating headings from their content, embedded with `nomic-embed-text`, and searched with cosine similarity in numpy.
- **Clients** (`evalh/clients/`): streaming clients for Ollama's native API and for any OpenAI-compatible endpoint. They measure time to first token, prefill and decode throughput from Ollama's own counters (prefill counts only tokens that were not served from the prompt cache), and never raise: every failure comes back as a result with a reason.
- **Runner** (`evalh/run.py`): loads one model at a time, unloads everything else first, warms up, then runs all items for each repeat with a different seeded order per repeat. It samples memory from Ollama (`size_vram`) and from the `llama-server` process, records machine, Ollama version and settings, git state and hashes of the inputs on every invocation, and can be resumed after an interruption. Server outages are not recorded as model failures; they stop the run so a resume redoes them.
- **Scoring** (`evalh/scoring/`, `evalh/score.py`): deterministic checks first (required facts with aliases, refusal on unanswerable questions, retrieval hit@4, tool name and argument matching with IP and ASN normalisation, invalid tool calls), then an LLM judge for correctness and faithfulness on a fixed rubric, with judgements cached by judge model and input.
- **Judge calibration** (`evalh/label.py`): a person labels a stratified sample of answers without seeing the model name or the judge's score, and the report gives Cohen's kappa between person and judge.
- **Report** (`evalh/report.py`): per-config tables for quality, latency (p50/p95, TTFT), throughput and memory, pass/fail against `configs/acceptance.yaml`, and a recommendation per use case: the smallest passing local config, else the hosted baseline, else none.
- **Gateway** (`gateway/`): `POST /v1/chat` with an ordered fallback chain per route. It falls back on errors, a first-token budget, a total timeout, or a tool call that fails its JSON schema; skips a backend after repeated failures (circuit breaker with a single half-open trial); and refuses to send routes marked `data_class: internal` to a hosted model, both at config load and at request time. Responses carry `x-served-by` and `x-fallback-reason`, and `/metrics` exposes Prometheus counters.

The design and the reasoning behind it are in `docs/design.md`. Live gateway transcripts are in `docs/gateway-transcripts.md`.

## How it scales

Data: the question set and corpus are small on purpose so that every gold answer can be checked by hand. Adding a use case means adding a JSONL file, a prompt builder and an acceptance block; the scorers and report do not change. The numpy index is fine to tens of thousands of chunks; beyond that the `Index` class is the one piece to replace with a vector database, behind the same `search` interface.

Compute: the runner deliberately runs one model at a time with a single Ollama slot, because the goal is clean per-model latency and memory numbers rather than throughput. Run time grows linearly with configs times items times repeats (about 10 to 12 hours for the full grid on an Apple M4). On a GPU server the same runner works unchanged against any OpenAI-compatible server such as vLLM or TGI by pointing the hosted client at it, and `--only` lets a new model be evaluated without re-running the rest.

Network: the gateway is async and holds no state except the circuit breaker, so it scales horizontally behind a load balancer; the breaker becomes per-instance, which is acceptable for its purpose. The hosted client retries rate limits with backoff and treats auth and rate-limit errors as outages rather than model failures.

Storage: results are append-only JSONL with one record per response plus provenance records, which keeps runs resumable and diffable. For many runs or many teams, the same records load directly into SQLite or Parquet.

What stays the same as load grows: the record format, the scorers, the acceptance file and the gateway's routing rules. What gets replaced: the inference server (Ollama to vLLM or TGI), the index, and the hosted baseline. What is configuration: `configs/models.yaml` (the grid), `configs/acceptance.yaml` (thresholds), `configs/routes.yaml` (gateway chains, timeouts and data classes).

## How to demo it

Requirements: Python 3.12 with [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com) running locally. The full grid needs about 55 GB of disk for the models.

```bash
uv sync
uv run pytest -q                      # 288 passed, 1 deselected (the live Ollama test)

# models
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b-instruct-q4_K_M    # the rest of the grid is listed in configs/models.yaml

# retrieval index (already committed; rebuild only if the corpus changes)
uv run python -m evalh.rag.build            # prints the chunk count per document

# a quick run on one small model, one repeat
uv run python -m evalh.run --only qwen2.5:1.5b-instruct-q4_K_M --repeats 1 --run-id demo
# output: results/raw-demo.jsonl; already done: 0 responses
# qwen2.5:1.5b-instruct-q4_K_M r1 rag-044: 2.1s ok
# ...

# score with a local judge (or set HOSTED_BASE_URL, HOSTED_API_KEY, HOSTED_MODEL and use --judge hosted)
uv run python -m evalh.score --run-id demo --judge qwen2.5:14b-instruct-q4_K_M

# report: reports/summary.md plus quality_vs_latency.png, memory.png, throughput.png
uv run python -m evalh.report --run-id demo

# optional: label answers yourself and see how well the judge agrees
uv run python -m evalh.label --run-id demo
uv run python -m evalh.label --run-id demo --report

# gateway
uv run uvicorn gateway.app:build_default_app --factory --port 8080
curl -s -i localhost:8080/v1/chat -H 'content-type: application/json' \
  -d '{"route":"doc-qa","messages":[{"role":"user","content":"What does BGP state Active mean?"}]}'
# response headers include x-served-by: qwen7b
```

The gateway also runs in Docker with `docker compose up`, talking to Ollama on the host. The hosted backend is used only when `HOSTED_BASE_URL`, `HOSTED_API_KEY` and `HOSTED_MODEL` are all set.

The full evaluation on an Apple M4 with 16 GB is in progress; its results, the methodology and the model selection guide will be added under `reports/` and `docs/` when it completes.
