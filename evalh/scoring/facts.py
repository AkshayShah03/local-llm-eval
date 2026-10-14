import re

_DASHES = re.compile(r"[‐-―\-_]")
# Case-sensitive on purpose: only an upper-case "AS" token is an ASN prefix, so the
# English word "as" ("known as 65535:666") is left alone. Lower-case "as2914" already
# matches the normalised form.
_ASN = re.compile(r"\bAS[\s_\-‐-―]+(\d+)")
_THOUSANDS = re.compile(r"(?<![\d,.])\d{1,3}(?:,\d{3})+(?!\d)")

# Bare "source"/"text" are too common in network prose ("the source does not have a
# route"), so they only count as the source material with a determiner like "provided".
_SRC = (r"(?:(?:provided|given|supplied) (?:sources?|text)|documents?|docs|documentation|"
        r"context|passages?|excerpts?|material)")
_DET = r"(?:the |these |those |provided |given )*"
_REFUSAL = re.compile(
    r"can['’]?t answer|cannot answer|(?:unable|not able) to answer|"
    r"\bi (?:don['’]t|do not) know\b|"
    r"\bi (?:don['’]t|do not) have (?:enough |sufficient )?information|"
    r"insufficient information|"
    rf"\b{_SRC} (?:do not|does not|don['’]t|doesn['’]t) "
    r"(?:contain|include|mention|provide|cover|say|specify|have (?:any |enough |sufficient )?information)|"
    r"\bnot (?:provided|mentioned|covered|included|specified|stated|present|found|available|addressed) "
    rf"in {_DET}{_SRC}|"
    r"\bno mention of\b|"
    rf"\bno (?:relevant )?information[^.]{{0,60}}\b{_SRC}|"
    rf"\b{_SRC}[^.]{{0,30}}\bno (?:relevant )?information|"
    rf"\b(?:could(?:n['’]t| not)|can['’]?t|cannot|unable to|not able to) find[^.]{{0,60}}\b{_SRC}"
)


def normalize(s: str) -> str:
    s = _ASN.sub(r"AS\1", s)  # before lowercasing; see _ASN
    s = _DASHES.sub(" ", s)
    s = _THOUSANDS.sub(lambda m: m.group().replace(",", ""), s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _contains(text: str, fact: str) -> bool:
    pattern = r"(?<![\w.:])" + re.escape(normalize(fact)) + r"(?!\w)(?![.:]\d)"
    return re.search(pattern, text) is not None


def fact_recall(text: str, facts: list) -> tuple[float, list]:
    if not facts:
        return 1.0, []
    norm = normalize(text)
    missing = [f for f in facts
               if not any(_contains(norm, alt) for alt in (f if isinstance(f, list) else [f]))]
    return (len(facts) - len(missing)) / len(facts), missing


def is_refusal(text: str) -> bool:
    return _REFUSAL.search(text.lower().replace("’", "'")) is not None
