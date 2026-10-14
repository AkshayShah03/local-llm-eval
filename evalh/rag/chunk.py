import re

# RFC plain-text pages end with a footer line and start with a header line.
_FOOTER = re.compile(r"^.*\[Page \d+\]\s*$", re.M)
_HEADER = re.compile(r"^RFC \d+ .*\d{4}\s*$", re.M)


def clean_rfc(text: str) -> str:
    text = text.replace("\f", "\n")
    text = _FOOTER.sub("", text)
    text = _HEADER.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_text(doc_id: str, text: str, max_words: int = 300) -> list[dict]:
    """Greedy paragraph packing; paragraphs longer than max_words are split on words."""
    pieces: list[str] = []
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        words = para.split()
        for i in range(0, len(words), max_words):
            pieces.append(" ".join(words[i:i + max_words]))

    chunks, current = [], []
    for piece in pieces:
        if current and sum(len(p.split()) for p in current) + len(piece.split()) > max_words:
            chunks.append("\n\n".join(current))
            current = []
        current.append(piece)
    if current:
        chunks.append("\n\n".join(current))
    return [{"id": f"{doc_id}-{n:03d}", "doc": doc_id, "text": t} for n, t in enumerate(chunks)]
