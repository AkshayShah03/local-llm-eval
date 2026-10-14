import re

_DASHES = re.compile(r"[‐-―\-_]")
_ASN = re.compile(r"\bas\s+(\d+)")

_REFUSAL = re.compile(
    r"can['’]?t answer|cannot answer|unable to answer|not in the provided|"
    r"(?:does not|doesn['’]t) (?:contain|include|mention|provide)|"
    r"no information|not (?:mentioned|covered|specified)|"
    r"(?:don['’]t|do not) have (?:enough )?information|insufficient information"
)


def normalize(s: str) -> str:
    s = _DASHES.sub(" ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return _ASN.sub(r"as\1", s)


def _contains(text: str, fact: str) -> bool:
    pattern = r"(?<![\w.])" + re.escape(normalize(fact)) + r"(?!\w)"
    return re.search(pattern, text) is not None


def fact_recall(text: str, facts: list) -> tuple[float, list]:
    if not facts:
        return 1.0, []
    norm = normalize(text)
    missing = [f for f in facts
               if not any(_contains(norm, alt) for alt in (f if isinstance(f, list) else [f]))]
    return (len(facts) - len(missing)) / len(facts), missing


def is_refusal(text: str) -> bool:
    return _REFUSAL.search(text.lower()) is not None
