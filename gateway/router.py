import json
import logging
from dataclasses import dataclass, field

from jsonschema import Draft202012Validator
from prometheus_client import Counter

from evalh.clients.base import ChatResult
from gateway.breaker import CircuitBreaker
from gateway.config import GatewayConfig, validate

log = logging.getLogger("gateway")
# Module-level so that building several Routers in one process does not re-register them.
FALLBACKS = Counter("gateway_fallbacks", "Backend attempts that fell through", ["route", "backend", "reason"])
SERVED = Counter("gateway_served", "Requests served", ["route", "backend"])


@dataclass
class RouteResult:
    result: ChatResult | None
    served_by: str | None
    attempts: list[tuple[str, str]] = field(default_factory=list)


def _tool_calls_valid(result: ChatResult, tools: list[dict]) -> bool:
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    for call in result.tool_calls:
        schema = schemas.get(call.get("name"))
        args = call.get("arguments")
        if schema is None or not isinstance(args, dict):
            return False
        if any(True for _ in Draft202012Validator(schema).iter_errors(args)):
            return False
    return True


def _reason(error: str) -> str:
    return error if error in ("timeout", "first_token_timeout") else "error"


class Router:
    def __init__(self, config: GatewayConfig, clients: dict, breaker: CircuitBreaker, check: bool = True):
        self.config = validate(config) if check else config
        self.clients = clients
        self.breaker = breaker

    async def chat(self, route_name: str, messages: list[dict], tools: list[dict] | None = None) -> RouteResult:
        route = self.config.routes[route_name]
        attempts = []
        for name in route.chain:
            backend = self.config.backends[name]
            reason = None
            if route.data_class == "internal" and backend.provider != "ollama":
                reason = "policy_blocked"
            elif name not in self.clients:
                reason = "error"  # e.g. hosted backend listed but not configured
            elif not self.breaker.allow(name):
                reason = "circuit_open"
            else:
                result = await self.clients[name].chat(
                    backend.model, messages, tools=tools, options={"temperature": 0},
                    first_token_timeout=route.first_token_timeout_s, total_timeout=route.total_timeout_s)
                if result.error is not None:
                    reason = _reason(result.error)
                elif tools and result.tool_calls and not _tool_calls_valid(result, tools):
                    reason = "invalid_tool_call"
                self.breaker.record(name, reason is None)
                if reason is None:
                    SERVED.labels(route_name, name).inc()
                    log.info(json.dumps({"route": route_name, "served_by": name,
                                         "attempts": len(attempts) + 1, "total_s": round(result.total_s, 3)}))
                    return RouteResult(result, name, attempts)
            FALLBACKS.labels(route_name, name, reason).inc()
            attempts.append((name, reason))
        log.warning(json.dumps({"route": route_name, "served_by": None, "attempts": attempts}))
        return RouteResult(None, None, attempts)
