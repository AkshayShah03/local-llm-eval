from evalh.rag.chunk import chunk_text, clean_rfc

RFC_PAGE = """Some text at the end of page one.

Rekhter, et al.             Standards Track                    [Page 5]
\f
RFC 4271                         BGP-4                      January 2006


Continues here on page two.
"""


def test_clean_rfc_strips_page_furniture():
    out = clean_rfc(RFC_PAGE)
    assert "[Page 5]" not in out and "\f" not in out and "January 2006" not in out
    assert "Some text at the end of page one." in out and "Continues here on page two." in out


def test_chunk_ids_are_stable_and_bounded():
    text = "\n\n".join(f"para {i} " + "word " * 120 for i in range(10))
    chunks = chunk_text("doc", text, max_words=300)
    assert [c["id"] for c in chunks][:2] == ["doc-000", "doc-001"]
    assert all(len(c["text"].split()) <= 300 for c in chunks)
    assert chunk_text("doc", text, max_words=300) == chunks


def test_oversized_paragraph_is_split():
    chunks = chunk_text("d", "word " * 700, max_words=300)
    assert len(chunks) == 3
