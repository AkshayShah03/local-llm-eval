import pytest

from evalh.clients.base import ChatResult
from evalh.scoring.judge import judge, judge_messages, parse_judgement
from evalh.scoring.kappa import cohen_kappa


def test_parse_judgement_tolerates_wrapping():
    text = 'Sure:\n```json\n{"correctness": 4, "faithfulness": 5, "reason": "ok"}\n```'
    assert parse_judgement(text) == {"correctness": 4, "faithfulness": 5, "reason": "ok"}


def test_parse_judgement_rejects_out_of_range():
    assert parse_judgement('{"correctness": 7, "faithfulness": 5}') is None
    assert parse_judgement("no json here") is None


def test_judge_prompt_contains_inputs():
    msgs = judge_messages("Q?", ["65535:666"], "ctx text", "ans text")
    joined = msgs[-1]["content"]
    assert "Q?" in joined and "65535:666" in joined and "ctx text" in joined and "ans text" in joined


class FakeClient:
    def __init__(self, text):
        self.text = text

    async def chat(self, model, messages, tools=None, options=None, **_):
        return ChatResult(content=self.text)


async def test_judge_calls_client():
    out = await judge(FakeClient('{"correctness": 3, "faithfulness": 2, "reason": "r"}'), "m", "q", [], "c", "a")
    assert out["correctness"] == 3


def test_kappa_known_values():
    assert cohen_kappa([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert cohen_kappa([1, 1, 2, 2], [1, 2, 2, 2]) == pytest.approx(0.5)
    assert cohen_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 4], labels=[1, 2, 3, 4, 5], weights="quadratic") > \
        cohen_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 4], labels=[1, 2, 3, 4, 5])
