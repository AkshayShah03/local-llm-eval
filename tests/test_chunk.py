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


TABLE = "Community | Local-Pref | Description\n(default) | 120 | customer\n65520:nnnn | 50 | country nnnn"


def test_multiline_paragraph_is_preserved_byte_for_byte():
    text = "Intro paragraph.\n\n" + TABLE + "\n\nTrailing paragraph."
    chunks = chunk_text("t", text)
    assert len(chunks) == 1
    assert TABLE in chunks[0]["text"]


def test_oversized_paragraph_splits_on_line_boundaries():
    lines = [f"row{i} | " + "w " * 9 for i in range(60)]
    chunks = chunk_text("t", "\n".join(lines), max_words=100)
    assert len(chunks) > 1
    assert all(len(c["text"].split()) <= 100 for c in chunks)
    rejoined = "\n".join(c["text"] for c in chunks).split("\n")
    assert rejoined == lines


def test_heading_is_not_orphaned_at_chunk_end():
    body = "word " * 250
    text = f"{body}\n\nEU MSA origins (2914:12-)\n\n" + "other " * 100
    chunks = chunk_text("h", text, max_words=300)
    assert len(chunks) == 2
    assert not chunks[0]["text"].rstrip().endswith("(2914:12-)")
    assert chunks[1]["text"].startswith("EU MSA origins (2914:12-)")
    assert all(len(c["text"].split()) <= 300 for c in chunks)


def test_heading_with_oversized_following_paragraph_stays_bounded():
    text = "Some Heading\n\n" + "\n".join(f"line{i} " + "w " * 19 for i in range(40))
    chunks = chunk_text("h", text, max_words=100)
    assert chunks[0]["text"].startswith("Some Heading")
    assert "line0" in chunks[0]["text"]
    assert all(len(c["text"].split()) <= 100 for c in chunks)


RFC_FRONT = """Abstract

   This document describes the use of a well-known community.

Status of This Memo

   This document is not an Internet Standards Track specification.

Copyright Notice

   Copyright (c) 2016 IETF Trust and the persons identified as the
   document authors.  All rights reserved.

Table of Contents

   1. Introduction ....................................................3
   2. Acknowledgements ................................................9
   Authors' Addresses .................................................9

1.  Introduction

   Body text about blackholing.

Authors' Addresses

   Thomas King
   Email: thomas.king@de-cix.net
"""


def test_clean_rfc_drops_front_matter_toc_and_addresses():
    out = clean_rfc(RFC_FRONT)
    for gone in ("Status of This Memo", "Copyright", "Table of Contents", "....", "Authors' Addresses",
                 "thomas.king", "Standards Track specification"):
        assert gone not in out
    assert "Abstract" in out and "well-known community" in out
    assert "1.  Introduction" in out and "Body text about blackholing." in out


def test_lead_in_sentence_ending_in_colon_stays_with_its_table():
    lead = "Customers may prepend to selected peers with the following communities, where nnn is the peer ASN used here today:"
    text = "word " * 280 + "\n\n" + lead + "\n\n" + TABLE
    chunks = chunk_text("l", text, max_words=300)
    assert len(chunks) == 2
    assert chunks[1]["text"].startswith(lead) and TABLE in chunks[1]["text"]
