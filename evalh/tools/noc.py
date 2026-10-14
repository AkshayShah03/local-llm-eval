"""Mock NOC toolset backed by a fixture file.

The schemas are what the model sees. The implementations exist so the gateway
demo and multi-turn experiments return realistic data; scoring only looks at
the calls the model makes.
"""
import ipaddress
import itertools
import json
from pathlib import Path


def _fn(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOLS = [
    _fn("get_bgp_session", "Get the current state of a BGP session by neighbor IP address.",
        {"neighbor": {"type": "string", "description": "Neighbor IPv4 or IPv6 address"}}, ["neighbor"]),
    _fn("lookup_prefix_origin", "Look up the origin AS and RPKI validation state of a prefix as seen in our routing table.",
        {"prefix": {"type": "string", "description": "Prefix in CIDR notation, e.g. 203.0.113.0/24"}}, ["prefix"]),
    _fn("get_interface_errors", "Get operational status and error counters for a router interface.",
        {"router": {"type": "string", "description": "Router hostname, e.g. r1.dal"},
         "interface": {"type": "string", "description": "Interface name, e.g. et-0/0/1"}},
        ["router", "interface"]),
    _fn("open_ticket", "Open a NOC ticket.",
        {"summary": {"type": "string"},
         "severity": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
         "router": {"type": "string", "description": "Affected router, if known"}},
        ["summary", "severity"]),
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


class NocTools:
    def __init__(self, fixtures: dict):
        self._f = fixtures
        self._ticket_seq = itertools.count(1001)

    @classmethod
    def from_file(cls, path: str | Path) -> "NocTools":
        return cls(json.loads(Path(path).read_text()))

    def call(self, name: str, args: dict) -> dict:
        handler = {
            "get_bgp_session": self._bgp,
            "lookup_prefix_origin": self._prefix,
            "get_interface_errors": self._iface,
            "open_ticket": self._ticket,
        }[name]
        return handler(**args)

    def _bgp(self, neighbor: str) -> dict:
        target = ipaddress.ip_address(neighbor.strip())
        for s in self._f["bgp_sessions"]:
            if ipaddress.ip_address(s["neighbor"]) == target:
                return s
        return {"error": f"no session for neighbor {neighbor}"}

    def _prefix(self, prefix: str) -> dict:
        target = ipaddress.ip_network(prefix.strip(), strict=False)
        for p in self._f["prefixes"]:
            if ipaddress.ip_network(p["prefix"]) == target:
                return p
        return {"error": f"prefix {prefix} not in table"}

    def _iface(self, router: str, interface: str) -> dict:
        for i in self._f["interfaces"]:
            if i["router"] == router and i["interface"] == interface:
                return i
        return {"error": f"no interface {interface} on {router}"}

    def _ticket(self, summary: str, severity: str, router: str | None = None) -> dict:
        return {"ticket_id": f"NOC-{next(self._ticket_seq)}", "severity": severity, "router": router}
