from pathlib import Path

from evalh.config import load_acceptance, load_grid
from evalh.data import append_jsonl, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def test_grid_loads_all_models():
    grid = load_grid(ROOT / "configs/models.yaml")
    local = [m for m in grid.models if m.provider == "ollama"]
    assert len(local) == 11
    assert len({m.name for m in grid.models}) == len(grid.models)
    assert grid.defaults["num_ctx"] == 8192
    assert grid.not_run[0]["name"] == "qwen2.5:14b-instruct-q8_0"


def test_acceptance_has_three_use_cases():
    acc = load_acceptance(ROOT / "configs/acceptance.yaml")
    assert set(acc) == {"noc_tools", "doc_qa", "triage"}
    assert acc["noc_tools"]["criteria"]["tool_selection"] == {"min": 0.90}


def test_jsonl_roundtrip(tmp_path):
    p = tmp_path / "x.jsonl"
    append_jsonl(p, {"a": 1})
    append_jsonl(p, {"a": 2})
    assert read_jsonl(p) == [{"a": 1}, {"a": 2}]
    assert read_jsonl(tmp_path / "missing.jsonl") == []
