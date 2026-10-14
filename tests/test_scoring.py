import pytest

from evalh.scoring.facts import fact_recall, is_refusal, normalize
from evalh.scoring.retrieval import hit_at_k
from evalh.scoring.stats import percentile
from evalh.scoring.toolcalls import score_tool_calls


def test_normalize_hyphens_and_asn():
    assert normalize("Local-Pref  is AS 2914") == "local pref is as2914"


def test_fact_recall_aliases_and_boundaries():
    text = "Use community 65535:666 and discard the route. Peer AS2914."
    recall, missing = fact_recall(text, ["65535:666", ["drop", "discard"], "AS 2914", "666:1"])
    assert recall == 0.75 and missing == ["666:1"]


def test_fact_recall_no_partial_number_match():
    assert fact_recall("timer is 180 seconds", ["80"])[0] == 0.0
    assert fact_recall("prefix 110.0.0.1", ["10.0.0.1"])[0] == 0.0


def test_refusal():
    assert is_refusal("I can't answer that from the provided documents.")
    assert is_refusal("The context does not contain information about pricing.")
    assert not is_refusal("The hold timer default is 90 seconds.")


def test_hit_at_k():
    assert hit_at_k(["a", "b", "c"], ["c"], k=3)
    assert not hit_at_k(["a", "b", "c"], ["c"], k=2)


def exp(name, **args):
    return {"name": name, "args": args}


def call(name, **args):
    return {"name": name, "arguments": args}


def test_tool_exact_match_with_normalisation():
    s = score_tool_calls([call("lookup_prefix_origin", prefix=" 203.0.113.0/24 ")],
                         [exp("lookup_prefix_origin", prefix="203.0.113.0/24")])
    assert s.selection_ok and s.args_ok and not s.invalid


def test_tool_parallel_any_order():
    s = score_tool_calls([call("lookup_prefix_origin", prefix="198.51.100.0/24"),
                          call("get_bgp_session", neighbor="198.51.100.9")],
                         [exp("get_bgp_session", neighbor="198.51.100.9"),
                          exp("lookup_prefix_origin", prefix="198.51.100.0/24")])
    assert s.selection_ok and s.args_ok


def test_tool_wrong_arg():
    s = score_tool_calls([call("get_bgp_session", neighbor="192.0.2.2")],
                         [exp("get_bgp_session", neighbor="192.0.2.1")])
    assert s.selection_ok and not s.args_ok


def test_ipv6_canonical_form():
    s = score_tool_calls([call("get_bgp_session", neighbor="2001:0db8:0000::5")],
                         [exp("get_bgp_session", neighbor="2001:db8::5")])
    assert s.args_ok


def test_abstention():
    assert score_tool_calls([], []).selection_ok
    assert not score_tool_calls([call("get_bgp_session", neighbor="192.0.2.1")], []).selection_ok


def test_invalid_calls():
    assert score_tool_calls([{"name": "get_bgp_session", "arguments": "{bad"}],
                            [exp("get_bgp_session", neighbor="192.0.2.1")]).invalid
    assert score_tool_calls([call("get_bgp_session")], [exp("get_bgp_session", neighbor="x")]).invalid
    assert score_tool_calls([call("reboot", x=1)], []).invalid
    text_call = '{"name": "get_bgp_session", "arguments": {"neighbor": "192.0.2.1"}}'
    assert score_tool_calls([], [exp("get_bgp_session", neighbor="192.0.2.1")], content=text_call).invalid


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
    assert percentile([], 50) is None
