from evalh.sysinfo import is_runner


def test_is_runner_matches_llama_server_and_legacy_runner():
    assert is_runner("llama-server", ["/x/llama-server", "--model", "m"])
    assert is_runner("ollama", ["/opt/ollama", "runner", "--model", "m"])
    assert not is_runner("ollama", ["/opt/ollama", "serve"])
    assert not is_runner("python", ["python", "x.py"])
