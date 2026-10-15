import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from evalh.clients.ollama import OllamaClient
from evalh.clients.openai_compat import OpenAICompatClient
from gateway.breaker import CircuitBreaker
from gateway.config import load_config
from gateway.router import BadRequest, Router


class ChatRequest(BaseModel):
    route: str
    messages: list[dict]
    tools: list[dict] | None = None


def create_app(router: Router) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            closed = set()
            for client in router.clients.values():
                if id(client) not in closed and hasattr(client, "aclose"):
                    closed.add(id(client))
                    await client.aclose()

    app = FastAPI(title="local-llm gateway", lifespan=lifespan)
    app.state.router = router

    @app.post("/v1/chat")
    async def chat(req: ChatRequest):
        if req.route not in router.config.routes:
            raise HTTPException(404, f"unknown route {req.route}")
        try:
            out = await router.chat(req.route, req.messages, req.tools)
        except BadRequest as e:
            raise HTTPException(400, str(e)) from e
        reasons = ",".join(r for _, r in out.attempts)
        if out.result is None:
            return JSONResponse({"error": "all backends failed", "attempts": out.attempts}, status_code=503,
                                headers={"x-fallback-reason": reasons})
        body = {"message": {"role": "assistant", "content": out.result.content, "tool_calls": out.result.tool_calls},
                "served_by": out.served_by}
        headers = {"x-served-by": out.served_by}
        if reasons:
            headers["x-fallback-reason"] = reasons
        return JSONResponse(body, headers=headers)

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    return app


def build_default_app() -> FastAPI:
    cfg = load_config(Path(os.environ.get("ROUTES_FILE", "configs/routes.yaml")))
    ollama = OllamaClient(os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    hosted_env = [os.environ.get(k) for k in ("HOSTED_BASE_URL", "HOSTED_API_KEY", "HOSTED_MODEL")]
    clients = {}
    for name, b in cfg.backends.items():
        if b.provider == "ollama":
            clients[name] = ollama
        elif all(hosted_env):
            clients[name] = OpenAICompatClient(*hosted_env)
    return create_app(Router(cfg, clients, CircuitBreaker()))
