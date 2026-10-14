import math
import warnings
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


# ---- Fix round 1: parsing, score validation, judge options, kappa edge cases ----

def test_parse_judgement_allows_braces_in_reason():
    text = '{"correctness": 4, "faithfulness": 5, "reason": "uses {x} syntax"}'
    assert parse_judgement(text) == {"correctness": 4, "faithfulness": 5, "reason": "uses {x} syntax"}


def test_parse_judgement_code_fenced_with_braces_in_reason():
    text = 'Here you go:\n```json\n{"correctness": 2, "faithfulness": 3, "reason": "f(){ return {} }"}\n```\nThanks.'
    assert parse_judgement(text) == {"correctness": 2, "faithfulness": 3, "reason": "f(){ return {} }"}


def test_parse_judgement_skips_invalid_object_before_valid_one():
    text = '{"note": "draft"} then {"correctness": 1, "faithfulness": 1, "reason": "r"}'
    assert parse_judgement(text) == {"correctness": 1, "faithfulness": 1, "reason": "r"}


def test_parse_judgement_rejects_bools():
    assert parse_judgement('{"correctness": true, "faithfulness": 5, "reason": "r"}') is None
    assert parse_judgement('{"correctness": 4, "faithfulness": false, "reason": "r"}') is None


def test_parse_judgement_accepts_integral_float():
    out = parse_judgement('{"correctness": 4.0, "faithfulness": 5.0, "reason": "r"}')
    assert out == {"correctness": 4, "faithfulness": 5, "reason": "r"}
    assert type(out["correctness"]) is int


def test_parse_judgement_rejects_non_integral_float():
    assert parse_judgement('{"correctness": 4.5, "faithfulness": 5, "reason": "r"}') is None


def test_parse_judgement_accepts_digit_string():
    out = parse_judgement('{"correctness": "4", "faithfulness": "5", "reason": "r"}')
    assert out == {"correctness": 4, "faithfulness": 5, "reason": "r"}
    assert type(out["faithfulness"]) is int


def test_parse_judgement_rejects_non_digit_string():
    assert parse_judgement('{"correctness": "four", "faithfulness": 5, "reason": "r"}') is None
    assert parse_judgement('{"correctness": "4.0", "faithfulness": 5, "reason": "r"}') is None


def test_parse_judgement_rejects_out_of_range_strings_and_floats():
    assert parse_judgement('{"correctness": "0", "faithfulness": 5}') is None
    assert parse_judgement('{"correctness": 6.0, "faithfulness": 5}') is None


class RecordingClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def chat(self, model, messages, tools=None, options=None, **_):
        self.calls.append({"model": model, "messages": messages, "options": options})
        return self.result


async def test_judge_passes_global_generation_options():
    client = RecordingClient(ChatResult(content='{"correctness": 5, "faithfulness": 5, "reason": "r"}'))
    await judge(client, "m", "q", ["f"], "c", "a")
    assert client.calls[0]["options"] == {"temperature": 0, "seed": 7, "num_ctx": 8192, "num_predict": 512}


async def test_judge_returns_none_on_client_error():
    client = RecordingClient(ChatResult.failed("timeout", 0.1))
    assert await judge(client, "m", "q", [], "c", "a") is None


async def test_judge_returns_none_on_unparseable_reply():
    client = RecordingClient(ChatResult(content="I cannot grade this."))
    assert await judge(client, "m", "q", [], "c", "a") is None


def test_kappa_quadratic_exact_value():
    a, b = [1, 2, 3, 4, 5], [1, 2, 3, 4, 4]
    assert cohen_kappa(a, b, labels=[1, 2, 3, 4, 5], weights="quadratic") == pytest.approx(16 / 17)


def test_kappa_empty_input_raises():
    with pytest.raises(ValueError):
        cohen_kappa([], [])


def test_kappa_label_outside_labels_raises_clear_error():
    with pytest.raises(ValueError, match="not in labels"):
        cohen_kappa([1, 2], [1, 3], labels=[1, 2])


def test_kappa_constant_same_label_is_nan():
    assert math.isnan(cohen_kappa([2, 2, 2], [2, 2, 2]))


def test_kappa_single_label_quadratic_is_nan_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert math.isnan(cohen_kappa([1, 1], [1, 1], labels=[1], weights="quadratic"))
