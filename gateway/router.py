import json
import logging
from dataclasses import dataclass, field

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
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


class BadRequest(ValueError):
    """The caller's tool definitions are unusable; no backend was tried."""


def _build_validators(tools: list[dict]) -> dict[str, Draft202012Validator]:
    try:
        validators = {}
        for t in tools:
            schema = t["function"]["parameters"]
            Draft202012Validator.check_schema(schema)
            validators[t["function"]["name"]] = Draft202012Validator(schema)
        return validators
    except (SchemaError, KeyError, TypeError) as e:
        raise BadRequest(f"invalid tool definition: {type(e).__name__}: {e}") from e


def _tool_calls_valid(result: ChatResult, validators: dict[str, Draft202012Validator]) -> bool:
    for call in result.tool_calls:
        validator = validators.get(call.get("name"))
        args = call.get("arguments")
        if validator is None or not isinstance(args, dict):
            return False
        if any(True for _ in validator.iter_errors(args)):
            return False
    return True


def _counts_against_backend(reason: str | None, error: str | None) -> bool:
    """Only failures that say something about backend health open the breaker.

    invalid_tool_call is a model-quality problem and an HTTP 4xx (other than 408/429) is a
    problem with the request or model name; in both cases the backend itself answered.
    """
    if reason is None or reason == "invalid_tool_call":
        return False
    if error and error.startswith("http 4") and not error.startswith(("http 408", "http 429")):
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
        validators = _build_validators(tools) if tools else {}
        attempts = []
        for name in route.chain:
            backend = self.config.backends[name]
            reason, error = None, None
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
                error = result.error
                if error is not None:
                    reason = _reason(error)
                elif tools and result.tool_calls and not _tool_calls_valid(result, validators):
                    reason = "invalid_tool_call"
                self.breaker.record(name, not _counts_against_backend(reason, error))
                if reason is None:
                    SERVED.labels(route_name, name).inc()
                    log.info(json.dumps({"route": route_name, "served_by": name,
                                         "attempts": len(attempts) + 1, "total_s": round(result.total_s, 3)}))
                    return RouteResult(result, name, attempts)
            FALLBACKS.labels(route_name, name, reason).inc()
            attempts.append((name, reason))
        log.warning(json.dumps({"route": route_name, "served_by": None, "attempts": attempts}))
        return RouteResult(None, None, attempts)
