import ipaddress
import json
import re
from dataclasses import dataclass

from evalh.tools.noc import TOOLS

_SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
_TEXT_CALL = re.compile(r'\{\s*"(?:name|function)"\s*:')


@dataclass
class ToolScore:
    selection_ok: bool
    args_ok: bool
    invalid: bool
    n_calls: int


def _norm(v):
    if isinstance(v, str):
        s = v.strip()
        try:
            return str(ipaddress.ip_interface(s)) if "/" in s else str(ipaddress.ip_address(s))
        except ValueError:
            pass
        if m := re.fullmatch(r"(?i)as\s*(\d+)", s):
            return int(m.group(1))
        if s.isdigit():
            return int(s)
        return s.lower()
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _parse(raw) -> dict | None:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return val if isinstance(val, dict) else None
    return None


def score_tool_calls(called: list[dict], expected: list[dict], content: str = "") -> ToolScore:
    invalid = False
    parsed = []
    for c in called:
        args = _parse(c.get("arguments"))
        schema = _SCHEMAS.get(c.get("name"))
        # Arguments must come from the tool schema (anything else is hallucinated). Extra
        # args that ARE in the schema but absent from the expected call (e.g. open_ticket
        # summary/router, free text) are allowed and simply not compared.
        if (args is None or schema is None
                or any(k not in args for k in schema.get("required", []))
                or any(k not in schema.get("properties", {}) for k in args)):
            invalid = True
            continue
        parsed.append((c["name"], args))
    # Some models write the call as JSON text instead of using the tool-call channel.
    if not called and _TEXT_CALL.search(content or ""):
        invalid = True

    selection_ok = not invalid and sorted(n for n, _ in parsed) == sorted(e["name"] for e in expected)
    args_ok = selection_ok and _all_matched(parsed, expected)
    return ToolScore(selection_ok=selection_ok, args_ok=args_ok, invalid=invalid, n_calls=len(called))


def _all_matched(parsed, expected) -> bool:
    unused = list(parsed)
    for e in expected:
        for i, (name, args) in enumerate(unused):
            if name == e["name"] and all(_norm(args.get(k)) == _norm(v) for k, v in e["args"].items()):
                unused.pop(i)
                break
        else:
            return False
    return True
