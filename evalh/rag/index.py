import json
from pathlib import Path

import numpy as np


class Index:
    def __init__(self, chunks: list[dict], vectors: np.ndarray):
        self.chunks = chunks
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = (vectors / np.where(norms == 0, 1, norms)).astype(np.float32)
        self._by_id = {c["id"]: c for c in chunks}

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[str, float]]:
        q = qvec / (np.linalg.norm(qvec) or 1)
        sims = self.vectors @ q.astype(np.float32)
        top = np.argsort(-sims)[:k]
        return [(self.chunks[i]["id"], float(sims[i])) for i in top]

    def get(self, chunk_id: str) -> dict:
        return self._by_id[chunk_id]

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", self.vectors)
        (d / "chunks.jsonl").write_text("".join(json.dumps(c) + "\n" for c in self.chunks))

    @classmethod
    def load(cls, directory: str | Path) -> "Index":
        d = Path(directory)
        chunks = [json.loads(l) for l in (d / "chunks.jsonl").read_text().splitlines() if l]
        return cls(chunks, np.load(d / "embeddings.npy"))
