from evalh.prompts import REFUSAL_SENTENCE, rag_messages, tool_messages, triage_messages


def test_rag_messages_include_passage_ids_and_refusal_rule():
    msgs = rag_messages("Q?", [{"id": "rfc4271-010", "text": "hold time"}])
    assert msgs[0]["role"] == "system" and REFUSAL_SENTENCE in msgs[0]["content"]
    assert "[rfc4271-010]" in msgs[1]["content"] and msgs[1]["content"].endswith("Question: Q?")


def test_tool_and_triage_messages():
    assert tool_messages("is it up?")[-1] == {"role": "user", "content": "is it up?"}
    assert "LOG LINE" in triage_messages("LOG LINE")[-1]["content"]
