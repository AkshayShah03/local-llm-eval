import re

# RFC plain-text pages end with a footer line and start with a header line.
_FOOTER = re.compile(r"^.*\[Page \d+\]\s*$", re.M)
_HEADER = re.compile(r"^RFC \d+ .*\d{4}\s*$", re.M)
# Unindented section headings whose content is boilerplate. Front-matter sections run until
# the next unindented line; address sections run to the end of the document.
_FRONT = re.compile(r"^(Status of This Memo|Copyright Notice|Table of Contents)\s*$")
_ADDRESSES = re.compile(r"^(Authors?|Editors?)['’]s?\s+Address(es)?\s*$")

HEADING_MAX_WORDS = 12
LEAD_IN_MAX_WORDS = 60


def _drop_boilerplate(text: str) -> str:
    out, skipping = [], False
    for line in text.split("\n"):
        if _ADDRESSES.match(line):
            break
        if _FRONT.match(line):
            skipping = True
            continue
        if skipping and line.strip() and not line[0].isspace():
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out)


def clean_rfc(text: str) -> str:
    text = text.replace("\f", "\n")
    text = _FOOTER.sub("", text)
    text = _HEADER.sub("", text)
    text = _drop_boilerplate(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _wc(text: str) -> int:
    return len(text.split())


def _is_heading(para: str) -> bool:
    s = para.strip()
    if "\n" in s:
        return False
    if s.endswith(":"):  # lead-in sentence for a list or table
        return _wc(s) <= LEAD_IN_MAX_WORDS
    return _wc(s) <= HEADING_MAX_WORDS and not s.endswith(".")


def _split(para: str, first_budget: int, max_words: int) -> list[str]:
    """Split a paragraph on line boundaries (words only for a single over-long line)."""
    pieces, cur, cur_words, budget = [], [], 0, first_budget
    for line in para.split("\n"):
        n = _wc(line)
        if n > budget:
            if cur:
                pieces.append("\n".join(cur))
                cur, cur_words = [], 0
                budget = max_words
            words = line.split()
            while len(words) > budget:
                pieces.append(" ".join(words[:budget]))
                words = words[budget:]
                budget = max_words
            if words:
                cur, cur_words = [" ".join(words)], len(words)
            continue
        if cur_words + n > budget:
            pieces.append("\n".join(cur))
            cur, cur_words, budget = [], 0, max_words
        cur.append(line)
        cur_words += n
    if cur:
        pieces.append("\n".join(cur))
    return pieces


def chunk_text(doc_id: str, text: str, max_words: int = 300) -> list[dict]:
    """Greedy paragraph packing.

    Paragraphs that fit are kept verbatim (newlines intact). Larger ones are split on line
    boundaries. A short heading-like paragraph is merged into the paragraph that follows it
    so it is never left at the end of a chunk.
    """
    pieces: list[str] = []
    heading = ""
    for para in text.split("\n\n"):
        para = para.strip("\n")
        if not para.strip():
            continue
        if _is_heading(para):
            heading = f"{heading}\n\n{para}" if heading else para
            continue
        if heading:
            combined = f"{heading}\n\n{para}"
            if _wc(combined) <= max_words:
                pieces.append(combined)
            else:
                pieces += _split(combined, max_words, max_words)
            heading = ""
        elif _wc(para) <= max_words:
            pieces.append(para)
        else:
            pieces += _split(para, max_words, max_words)
    if heading:
        pieces.append(heading)

    chunks, current, size = [], [], 0
    for piece in pieces:
        n = _wc(piece)
        if current and size + n > max_words:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        current.append(piece)
        size += n
    if current:
        chunks.append("\n\n".join(current))
    return [{"id": f"{doc_id}-{n:03d}", "doc": doc_id, "text": t} for n, t in enumerate(chunks)]
