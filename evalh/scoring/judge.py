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

_OBJ = re.compile(r"\{.*?\}", re.S)


def judge_messages(question: str, reference_facts: list, context: str, answer: str) -> list[dict]:
    facts = "\n".join(f"- {' / '.join(f) if isinstance(f, list) else f}" for f in reference_facts) or "- (none given)"
    return [{"role": "user", "content": RUBRIC.format(
        question=question, facts=facts, context=context, answer=answer or "(empty)")}]


def parse_judgement(text: str) -> dict | None:
    for match in _OBJ.finditer(text or ""):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        c, f = obj.get("correctness"), obj.get("faithfulness")
        if isinstance(c, int) and isinstance(f, int) and 1 <= c <= 5 and 1 <= f <= 5:
            return {"correctness": c, "faithfulness": f, "reason": str(obj.get("reason", ""))}
    return None


async def judge(client, model: str, question: str, reference_facts: list, context: str, answer: str) -> dict | None:
    result = await client.chat(model, judge_messages(question, reference_facts, context, answer),
                               options={"temperature": 0, "seed": 7, "num_predict": 200})
    return parse_judgement(result.content) if result.ok else None
