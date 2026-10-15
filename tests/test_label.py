import json

from evalh.data import read_jsonl
from evalh.label import agreement, main, sample_for_labelling


def scored(config, item, c, **extra):
    return {"type": "response", "config": config, "item_id": item, "repeat": 1, "task": "rag",
            "result": {"content": f"answer for {item}"},
            "scores": {"failed": False, "correctness": c, **extra}}


def test_sample_is_stratified_and_deterministic():
    rows = [scored(m, f"q{i}", 3) for m in ("a", "b", "c") for i in range(20)]
    s = sample_for_labelling(rows, n=9)
    assert [r["config"] for r in s].count("a") == 3
    assert s == sample_for_labelling(rows, n=9)


def test_sample_skips_unjudged_and_repeats():
    rows = [scored("a", "q1", 4),
            {**scored("a", "q2", 4), "repeat": 2},
            {"type": "response", "config": "a", "item_id": "q3", "repeat": 1, "scores": {"failed": True}},
            {"type": "response", "config": "a", "item_id": "q4", "repeat": 1, "scores": {"failed": False, "judge_failed": True}},
            {"type": "meta", "config": "a"}]
    assert [r["item_id"] for r in sample_for_labelling(rows, n=10)] == ["q1"]


def test_agreement():
    rows = [scored("a", "q1", 5), scored("a", "q2", 2), scored("a", "q3", 4)]
    labels = [{"config": "a", "item_id": "q1", "correctness": 5},
              {"config": "a", "item_id": "q2", "correctness": 1},
              {"config": "a", "item_id": "q3", "correctness": 4}]
    out = agreement(labels, rows)
    assert out["n"] == 3 and out["kappa_binary"] == 1.0 and out["exact_agreement"] == 2 / 3


def test_agreement_empty_and_undefined():
    assert agreement([], [scored("a", "q1", 5)]) == {
        "n": 0, "kappa_binary": None, "kappa_quadratic": None, "exact_agreement": None}
    # every rating is the same on both sides, so kappa is undefined; JSON must stay NaN-free
    out = agreement([{"config": "a", "item_id": "q1", "correctness": 5}], [scored("a", "q1", 5)])
    assert out["n"] == 1 and out["kappa_binary"] is None and out["exact_agreement"] == 1.0
    json.dumps(out, allow_nan=False)


def test_cli_labels_two_items_and_report(tmp_path, capsys):
    root = tmp_path
    (root / "results").mkdir()
    (root / "data/questions").mkdir(parents=True)
    rows = [scored("model-x", "rag-1", 5), scored("model-y", "rag-2", 2)]
    (root / "results/scored-r1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "data/questions/rag.jsonl").write_text("".join(json.dumps(q) + "\n" for q in [
        {"id": "rag-1", "question": "Q one?", "required_facts": [["90", "ninety"]]},
        {"id": "rag-2", "question": "Q two?", "required_facts": [["1/3"]]}]))
    labels = tmp_path / "labels.jsonl"
    answers = iter(["x", "5", "1"])  # first prompt rejects "x" and re-asks
    main(["--run-id", "r1", "--n", "2"], input_fn=lambda _p: next(answers), labels_path=labels, root=root)
    out = capsys.readouterr().out
    assert "model-x" not in out and "model-y" not in out
    assert "90 | ninety" in out
    got = {(l["config"], l["item_id"]): l["correctness"] for l in read_jsonl(labels)}
    assert got == {("model-x", "rag-1"): 5, ("model-y", "rag-2"): 1}

    main(["--run-id", "r1", "--report"], labels_path=labels, root=root)
    report = capsys.readouterr().out
    assert "n: 2" in report and "kappa_binary: 1.000" in report
