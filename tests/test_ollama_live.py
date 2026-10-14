import pytest

from evalh.clients.ollama import OllamaClient


@pytest.mark.ollama
async def test_small_model_answers():
    async with OllamaClient() as c:
        r = await c.chat("qwen2.5:1.5b-instruct-q4_K_M",
                         [{"role": "user", "content": "Say OK."}],
                         options={"temperature": 0, "num_predict": 8})
    assert r.ok and r.content and r.decode_tps
