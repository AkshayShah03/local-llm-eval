import numpy as np

from evalh.rag.index import Index


def _index():
    chunks = [{"id": "a-000", "doc": "a", "text": "x"}, {"id": "a-001", "doc": "a", "text": "y"},
              {"id": "b-000", "doc": "b", "text": "z"}]
    vecs = np.array([[1, 0, 0], [0.7, 0.7, 0], [0, 0, 1]], dtype=np.float32)
    return Index(chunks, vecs)


def test_search_orders_by_cosine():
    hits = _index().search(np.array([1, 0.1, 0], dtype=np.float32), k=2)
    assert [h[0] for h in hits] == ["a-000", "a-001"]
    assert hits[0][1] > hits[1][1]


def test_save_load_roundtrip(tmp_path):
    idx = _index()
    idx.save(tmp_path)
    again = Index.load(tmp_path)
    assert again.get("b-000")["text"] == "z"
    assert again.search(np.array([0, 0, 1.0]), k=1)[0][0] == "b-000"
