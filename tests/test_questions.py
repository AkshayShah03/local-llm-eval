import json
from pathlib import Path

import pytest

from evalh.data import read_jsonl
from evalh.scoring.facts import fact_recall
from evalh.tools.noc import TOOLS

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = {c["id"]: c for c in read_jsonl(ROOT / "data/index/chunks.jsonl")}
RAG = read_jsonl(ROOT / "data/questions/rag.jsonl")
TOOLQ = read_jsonl(ROOT / "data/questions/tools.jsonl")
TRIAGE = read_jsonl(ROOT / "data/questions/triage.jsonl")
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}


def test_sizes():
    assert len(RAG) >= 30 and len(TOOLQ) >= 20 and len(TRIAGE) >= 10
    assert sum(not q["answerable"] for q in RAG) >= 6
    assert sum(q["category"] == "no-tool" for q in TOOLQ) >= 4
    ids = [q["id"] for q in RAG + TOOLQ + TRIAGE]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("q", [q for q in RAG if q["answerable"]], ids=lambda q: q["id"])
def test_rag_facts_are_in_gold_chunks(q):
    assert q["gold_chunk_ids"] and q["required_facts"]
    gold_text = "\n".join(CHUNKS[c]["text"] for c in q["gold_chunk_ids"])
    recall, missing = fact_recall(gold_text, q["required_facts"])
    assert recall == 1.0, f"facts not in gold chunks: {missing}"


@pytest.mark.parametrize("q", TOOLQ, ids=lambda q: q["id"])
def test_expected_calls_match_schemas(q):
    for call in q["expected_calls"]:
        props = SCHEMAS[call["name"]]["properties"]
        assert set(call["args"]) <= set(props)


@pytest.mark.parametrize("q", TRIAGE, ids=lambda q: q["id"])
def test_triage_facts_are_in_log(q):
    assert fact_recall(q["log"], q["required_facts"])[0] == 1.0
