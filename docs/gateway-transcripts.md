# Gateway live transcripts

These are two real requests made against the gateway running locally (uvicorn on port 8080) in front of a local Ollama instance on the test machine. The request body is the `noc-assistant` route with the NOC tool definitions from `evalh/tools/noc.py` and the user message "Is the session to 198.51.100.9 up?". Response bodies are shown as returned; the `date` header and `content-length` are kept, the `server` header is dropped.

Timeout note: with the shipped `configs/routes.yaml` (`first_token_timeout_s: 3` on `noc-assistant`) both requests below returned 503 on this machine, because the first token took longer than 3 seconds even with the model resident (a direct client call measured 3.7 s time to first token warm, 15.5 s cold). The two transcripts therefore use a copy of the routes file with the `noc-assistant` timeouts raised to 30 s first token and 60 s total; everything else in the file is unchanged. The 3 s default needs revisiting against the latency results before it is relied on.

## 1. Normal route

Command: `curl -s -i localhost:8080/v1/chat -d @req.json -H 'content-type: application/json'`

```
HTTP/1.1 200 OK
date: Thu, 15 Oct 2026 02:11:09 GMT
x-served-by: qwen7b
content-length: 148
content-type: application/json

{"message":{"role":"assistant","content":"","tool_calls":[{"name":"get_bgp_session","arguments":{"neighbor":"198.51.100.9"}}]},"served_by":"qwen7b"}

```

The primary backend (`qwen7b`) served the request; no `x-fallback-reason` header is present.

## 2. Fallback

The routes file for this run points `qwen7b` at a model name that does not exist (`qwen2.5:does-not-exist`). Ollama returns an HTTP error for it, the router records reason `error` and moves to `llama8b`.

```
HTTP/1.1 200 OK
date: Thu, 15 Oct 2026 02:11:29 GMT
x-served-by: llama8b
x-fallback-reason: error
content-length: 149
content-type: application/json

{"message":{"role":"assistant","content":"","tool_calls":[{"name":"get_bgp_session","arguments":{"neighbor":"198.51.100.9"}}]},"served_by":"llama8b"}

```

The metrics endpoint afterwards showed:

```
gateway_fallbacks_total{backend="qwen7b",reason="error",route="noc-assistant"} 1.0
gateway_served_total{backend="llama8b",route="noc-assistant"} 1.0
```
