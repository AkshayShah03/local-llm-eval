from pathlib import Path

import pytest

from evalh.tools.noc import TOOL_NAMES, TOOLS, NocTools

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def noc():
    return NocTools.from_file(ROOT / "data/fixtures/noc.json")


def test_schemas_are_function_format():
    assert TOOL_NAMES == {"get_bgp_session", "lookup_prefix_origin", "get_interface_errors", "open_ticket"}
    for t in TOOLS:
        assert t["type"] == "function"
        assert t["function"]["parameters"]["type"] == "object"


def test_bgp_session_lookup(noc):
    assert noc.call("get_bgp_session", {"neighbor": "198.51.100.9"})["state"] == "Idle"
    assert noc.call("get_bgp_session", {"neighbor": "10.9.9.9"}) == {"error": "no session for neighbor 10.9.9.9"}


def test_prefix_origin_normalises_input(noc):
    r = noc.call("lookup_prefix_origin", {"prefix": "198.51.100.0/24"})
    assert r["origin_asn"] == 64511 and r["rpki"] == "invalid"


def test_interface_errors(noc):
    r = noc.call("get_interface_errors", {"router": "r2.lax", "interface": "et-0/0/3"})
    assert r["crc_errors"] == 15234


def test_open_ticket(noc):
    r = noc.call("open_ticket", {"summary": "x", "severity": "P2"})
    assert r["ticket_id"].startswith("NOC-")


def test_unknown_tool(noc):
    with pytest.raises(KeyError):
        noc.call("reboot_router", {})
