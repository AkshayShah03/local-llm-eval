"""Chunk the corpus and embed it with nomic-embed-text. Usage: python -m evalh.rag.build"""
import asyncio
import json
from pathlib import Path

from evalh.clients.ollama import OllamaClient
from evalh.rag.chunk import chunk_text, clean_rfc
from evalh.rag.index import Index

ROOT = Path(__file__).resolve().parents[2]
EMBED_MODEL = "nomic-embed-text"


async def main() -> None:
    manifest = json.loads((ROOT / "data/corpus/manifest.json").read_text())
    chunks = []
    per_doc = {}
    for doc in manifest:
        text = (ROOT / "data/corpus" / f"{doc['id']}.txt").read_text()
        if doc["id"].startswith("rfc"):
            text = clean_rfc(text)
        doc_chunks = chunk_text(doc["id"], text)
        per_doc[doc["id"]] = len(doc_chunks)
        chunks += doc_chunks
    async with OllamaClient() as client:
        vectors = await client.embed(EMBED_MODEL, ["search_document: " + c["text"] for c in chunks])
    Index(chunks, vectors).save(ROOT / "data/index")
    for doc_id, n in per_doc.items():
        print(f"  {doc_id}: {n}")
    print(f"{len(chunks)} chunks from {len(manifest)} documents")


if __name__ == "__main__":
    asyncio.run(main())
