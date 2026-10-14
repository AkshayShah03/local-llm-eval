REFUSAL_SENTENCE = "I can't answer that from the provided documents."

RAG_SYSTEM = (
    "You are a network engineering assistant. Answer using only the context passages. "
    "Cite the passage ids you used in square brackets, for example [rfc4271-031]. "
    f"If the passages do not contain the answer, reply exactly: {REFUSAL_SENTENCE}"
)
TOOLS_SYSTEM = (
    "You are a NOC assistant for an IP backbone. Use the tools when a question needs live "
    "network state. Do not call tools for general questions you can answer directly. "
    "If several lookups are needed, call all of them."
)
TRIAGE_SYSTEM = (
    "Summarise this incident log for the on-call engineer in at most five bullet points: "
    "affected router and peer or interface, prefixes involved, when it started (UTC), "
    "the likely cause, and a suggested next step."
)


def rag_messages(question: str, passages: list[dict]) -> list[dict]:
    context = "\n\n".join(f"[{p['id']}]\n{p['text']}" for p in passages)
    return [{"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {question}"}]


def tool_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": TOOLS_SYSTEM}, {"role": "user", "content": question}]


def triage_messages(log: str) -> list[dict]:
    return [{"role": "system", "content": TRIAGE_SYSTEM}, {"role": "user", "content": log}]
