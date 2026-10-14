import json
import re

RUBRIC = """You are grading an answer written by a network-engineering assistant.

Question:
{question}

Reference facts (a fully correct answer contains all of these):
{facts}

Source material the assistant was given:
{context}

Answer to grade:
{answer}

Score two things from 1 to 5.
correctness: 5 = every reference fact present and nothing wrong; 4 = minor omission;
3 = partly correct or missing an important fact; 2 = mostly wrong; 1 = wrong or no answer.
faithfulness: 5 = every claim is supported by the source material; 3 = some unsupported claims;
1 = mostly unsupported or contradicts the source.

Reply with JSON only: {{"correctness": <1-5>, "faithfulness": <1-5>, "reason": "<one sentence>"}}"""


def judge_messages(question: str, reference_facts: list, context: str, answer: str) -> list[dict]:
    facts = "\n".join(f"- {' / '.join(f) if isinstance(f, list) else f}" for f in reference_facts) or "- (none given)"
    return [{"role": "user", "content": RUBRIC.format(
        question=question, facts=facts, context=context, answer=answer or "(empty)")}]


_DECODER = json.JSONDecoder()
_DIGITS = re.compile(r"\d+")


def _score(value) -> int | None:
    """Return value as an int in 1..5, or None. Bools are rejected."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        n = value
    elif isinstance(value, float):
        if not value.is_integer():
            return None
        n = int(value)
    elif isinstance(value, str) and _DIGITS.fullmatch(value.strip()):
        n = int(value.strip())
    else:
        return None
    return n if 1 <= n <= 5 else None


def parse_judgement(text: str) -> dict | None:
    """Return the first JSON object in text with valid scores, or None."""
    for pos, ch in enumerate(text or ""):
        if ch != "{":
            continue
        try:
            obj, _ = _DECODER.raw_decode(text, pos)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        c, f = _score(obj.get("correctness")), _score(obj.get("faithfulness"))
        if c is not None and f is not None:
            return {"correctness": c, "faithfulness": f, "reason": str(obj.get("reason", ""))}
    return None


async def judge(client, model: str, question: str, reference_facts: list, context: str, answer: str) -> dict | None:
    result = await client.chat(model, judge_messages(question, reference_facts, context, answer),
                               options={"temperature": 0, "seed": 7, "num_ctx": 8192, "num_predict": 512})
    return parse_judgement(result.content) if result.ok else None
