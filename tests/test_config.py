# tests/test_config.py
from pathlib import Path

def test_defaults(monkeypatch):
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("OLLAMA_BRAIN_PORT", raising=False)
    monkeypatch.delenv("OLLAMA_BRAIN_DATA_DIR", raising=False)
    import importlib, config
    importlib.reload(config)
    s = config.Settings()
    assert s.ollama_url == "http://localhost:11434"
    assert s.embed_model == "nomic-embed-text"
    assert s.chat_model == "llama3.2"
    assert s.server_port == 11435
    assert s.data_dir == Path.home() / ".ollama-brain"

def test_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:9999")
    monkeypatch.setenv("OLLAMA_BRAIN_PORT", "12000")
    import importlib, config
    importlib.reload(config)
    s = config.Settings()
    assert s.ollama_url == "http://localhost:9999"
    assert s.server_port == 12000
