# Design: local LLM evaluation harness

Status: agreed 2026-10-14, before any code or results.

## Goal

Decide, with evidence, which locally hosted open model (if any) should serve each
network-operations use case, and at what quantization, compared with a hosted API.
Output is a reproducible harness, raw results, a model-selection guide with
acceptance criteria, and a small gateway that implements the recommendation with
fallback.

Test machine: Apple M4, 16 GB unified memory. On Apple Silicon "GPU memory" is
unified memory; it is reported as such, measured from Ollama's `size_vram`.

## Use cases and test set

Domain is IP backbone operations (AS 2914 context), all from public sources.

- **RAG / doc Q&A**: ~30 hand-written questions over a small corpus: IETF RFC
  excerpts (4271 BGP-4, 1997 and 8092 communities, 6811 RPKI origin validation)
  and NTT's public AS2914 routing-policy / BGP-community pages. Each corpus file
  records its source URL and fetch date. Each question has `required_facts[]`,
  `gold_chunk_ids[]`, and `answerable` (some questions are deliberately not
  answerable from the corpus, to test refusal).
- **Tool calling**: ~20 questions against a mock NOC toolset
  (`get_bgp_session`, `lookup_prefix_origin`, `get_interface_errors`,
  `open_ticket`) backed by a consistent fixture file. Each question has
  `expected_calls[{name, args}]`, or `[]` when no tool should be called.
- **Offline triage summaries**: ~10 synthetic incident logs (syslog/BGP flap
  excerpts consistent with the fixtures), each with `required_facts[]` (affected
  peer, prefix, start time, likely cause). Scored by fact recall and the judge,
  relative to the hosted baseline.

Questions are reviewed by hand before any model is run.

## Models

| Model        | Quantizations                         |
|--------------|---------------------------------------|
| qwen2.5 1.5B | q4_K_M, q8_0                          |
| qwen2.5 3B   | q4_K_M, q8_0                          |
| qwen2.5 7B   | q4_K_M, q8_0                          |
| qwen2.5 14B  | q4_K_M (q8_0 does not fit in 16 GB; documented, not hidden) |
| llama3.1 8B  | q4_K_M, q8_0                          |
| mistral 7B   | q4_K_M, q8_0                          |

Hosted baseline: any OpenAI-compatible endpoint (OpenAI, Groq, Together) via
`base_url` + key in `.env`. Local runs do not need a key.

Embeddings: `nomic-embed-text` via Ollama; numpy cosine top-k over ~400-token chunks.

## Layout

```
data/corpus, data/questions/{rag,tools}.jsonl, data/fixtures/noc.json
evalh/clients/{ollama,openai_compat}.py
evalh/rag/index.py
evalh/tools/noc.py
evalh/scoring/{facts,retrieval,toolcalls,judge}.py
evalh/runner.py, evalh/report.py
gateway/
configs/models.yaml, configs/acceptance.yaml
docs/{reference-design,model-selection-guide,methodology}.md
results/   (raw JSONL, committed)
tests/
```

Clients know nothing about scoring; scorers are pure functions over a result
record; the runner only orchestrates; the gateway reuses the clients.

## Measurement

- TTFT: wall clock, request sent to first streamed token.
- Decode tok/s: `eval_count / eval_duration`. Prefill tok/s:
  `prompt_eval_count / prompt_eval_duration`.
- End-to-end latency p50/p95 per config per task.
- Controls: one uncounted warm-up per model load (cold-load time recorded
  separately), `temperature=0`, fixed `seed`, `num_ctx=8192` for every model,
  3 repeats per question (quality scored on repeat 1, latency over all repeats).
- Memory: `/api/ps` `size` and `size_vram`, plus sampled peak RSS of the Ollama
  runner via psutil. KV cache at 8k context reported separately from weights;
  the estimate `params × bits/8 + KV + overhead` is checked against measurement.
- Every run records chip, RAM, OS version, Ollama version, model digest, git SHA.

## Scoring

- Fact recall: fraction of `required_facts` present (normalised match).
- Retrieval hit@k against `gold_chunk_ids`.
- Refusal rate on unanswerable questions.
- LLM judge (hosted model) scores faithfulness and correctness 1–5 with a fixed
  rubric. Calibrated against ~30 hand labels; Cohen's κ reported.
- Tool calls: correct tool selected, exact-argument match (after normalisation),
  correct abstention when no tool is expected, invalid-JSON rate.
- Failures and timeouts are recorded with a reason, never dropped.

## Acceptance criteria

Fixed in `configs/acceptance.yaml` and committed before results exist.

| Use case | Criteria |
|---|---|
| NOC tool-calling assistant | tool selection ≥ 90%, exact args ≥ 85%, invalid JSON ≤ 2%, p95 ≤ 4 s |
| Doc Q&A (RAG) | fact recall ≥ 80%, faithfulness ≥ 4.0/5, refusal on unanswerable ≥ 80%, p95 ≤ 6 s |
| Offline bulk summarise/triage | quality ≥ baseline − 10%, cost per 1k requests reported, no latency gate |

Recommendation per use case: cheapest config that passes, or "use hosted" if none
does, with privacy, cost (stated assumptions: amortised hardware, power vs. per-token
price) and hardware fit (16 GB vs 24 GB GPU) discussed.

## Gateway

- `POST /v1/chat`, OpenAI-style body plus a `route` name. Routes map to an
  ordered model list in config.
- Fall back to the next model on: connection error / 5xx, TTFT budget exceeded,
  total timeout, or a tool call that is invalid JSON or fails the tool schema.
- Response headers `x-served-by`, `x-fallback-reason`; one JSON log line per
  request; `/metrics` with Prometheus counters by fallback reason.
- Circuit breaker: after N consecutive failures a model is skipped for 30 s.
- Hosted fallback is opt-in per route. A route with `data_class: internal` must
  never reach the hosted model; enforced by a test.
- `docker compose up` with Ollama on the host.

## Harness behaviour

- Resumable: skips (config, question, repeat) already present in the JSONL.
- Report script produces tables and charts into `reports/` and the numbers used
  in the selection guide.

## Testing

pytest, no Ollama required (HTTP mocked with respx): scorer edge cases
(argument normalisation, no-call case), percentile maths, deterministic retrieval
on fixed embeddings, gateway failover per reason, circuit breaker, internal-route
privacy rule. One optional `@pytest.mark.ollama` smoke test against a real model.
CI runs everything except the Ollama-marked test.

## Out of scope

Web UI, LangChain, vector database, fine-tuning. Listed under "next steps" in the
README instead.

## Build order

1. Corpus and questions (human-reviewed).
2. Clients and scorers, test-first.
3. Runner.
4. Pull models, run grid (~2–4 h machine time).
5. Hand-label ~30 answers, calibrate judge.
6. Report and docs.
7. Gateway.
