from evalh.run import done_keys, is_unexpected_error, order_items


def test_done_keys_only_counts_responses():
    recs = [{"type": "run_meta"},
            {"type": "response", "config": "m", "task": "rag", "item_id": "rag-001", "repeat": 1}]
    assert done_keys(recs) == {("m", "rag", "rag-001", 1)}


def test_order_items_is_seeded_and_differs_per_repeat():
    items = [{"id": str(i)} for i in range(20)]
    a1, a2 = order_items(items, repeat=1), order_items(items, repeat=2)
    assert a1 == order_items(items, repeat=1)
    assert [x["id"] for x in a1] != [x["id"] for x in a2]
    assert sorted(x["id"] for x in a1) == sorted(x["id"] for x in items)


def test_is_unexpected_error():
    assert not is_unexpected_error(None)
    assert not is_unexpected_error("timeout")
    assert not is_unexpected_error("first_token_timeout")
    assert not is_unexpected_error("http 500: boom")
    assert is_unexpected_error("KeyError: 'x'")
