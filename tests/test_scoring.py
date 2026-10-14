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


# --- fix round 1 regression tests ---

@pytest.mark.parametrize("text,fact", [
    ("set the timer as 90 seconds", "90"),
    ("the community known as 65535:666", "65535:666"),
    ("Peer AS 2914 is up", "AS2914"),
    ("Peer AS2914 is up", "AS 2914"),
    ("Peer AS-2914 is up", "AS 2914"),
    ("Peer AS_2914 is up", "AS 2914"),
    ("peer as2914 is up", "AS 2914"),
    ("The hold timer is 90.", "90"),
    ("Use 65535:666, then drop.", "65535:666"),
    ("Value: 90, then", "90"),
    ("limit is 1,000 routes", "1000"),
    ("limit is 1000 routes", "1,000"),
    ("prefix 10.0.0.0/8 here", "10.0.0.0/8"),
])
def test_fact_matches(text, fact):
    assert fact_recall(text, [fact])[0] == 1.0


@pytest.mark.parametrize("text,fact", [
    ("community 65535:666", "666"),
    ("wait 1.5 seconds", "1"),
    ("ratio 3:1", "1"),
    ("limit is 11,000 routes", "1000"),
])
def test_fact_no_match(text, fact):
    assert fact_recall(text, [fact])[0] == 0.0


@pytest.mark.parametrize("text", [
    "I can't answer that from the provided documents.",
    "The context does not contain information about pricing.",
    "The provided documents do not contain information about pricing.",
    "The documents don't mention pricing.",
    "I don't know based on the context.",
    "That information is not provided in the context.",
    "There is no mention of pricing in the passages.",
    "I cannot answer that.",
    "There is no information about pricing in the documents.",
    "The provided context has no information about pricing.",
    "I could not find this in the provided documents.",
    "I am not able to answer from the context.",
    "The answer is not stated in the context.",
])
def test_refusal_positive(text):
    assert is_refusal(text)


@pytest.mark.parametrize("text", [
    "BGP does not provide authentication by default",
    "If the MED is not specified, it is treated as 0",
    "The default route does not include the blackhole community",
    "no information is lost",
    "The RIB doesn't contain the prefix",
    "The hold timer default is 90 seconds.",
    "If the source does not have a valid route, the packet is dropped.",
    "The update text doesn't include the community.",
    "The router could not find a valid route.",
])
def test_refusal_negative(text):
    assert not is_refusal(text)


def test_tool_host_bits_must_match():
    s = score_tool_calls([call("lookup_prefix_origin", prefix="203.0.113.77/24")],
                         [exp("lookup_prefix_origin", prefix="203.0.113.0/24")])
    assert s.selection_ok and not s.args_ok


def test_tool_hallucinated_arg_invalid():
    s = score_tool_calls([call("get_bgp_session", neighbor="192.0.2.1", vrf="x")],
                         [exp("get_bgp_session", neighbor="192.0.2.1")])
    assert s.invalid and not s.selection_ok


def test_tool_extra_schema_arg_allowed():
    s = score_tool_calls([call("open_ticket", summary="s", router="r1", severity="low")],
                         [exp("open_ticket", severity="low")])
    assert s.args_ok and not s.invalid
