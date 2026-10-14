def hit_at_k(retrieved: list[str], gold: list[str], k: int) -> bool:
    return bool(set(retrieved[:k]) & set(gold))
