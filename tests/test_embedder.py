import pytest
import respx
import httpx


@pytest.mark.asyncio
async def test_embed_returns_vector(tmp_data_dir, monkeypatch):
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    import importlib
    import config
    importlib.reload(config)
    import embedder
    importlib.reload(embedder)
    fake_vector = [0.1] * 768
    with respx.mock:
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": fake_vector})
        )
        result = await embedder.embed("hello world")
    assert result == fake_vector
    assert len(result) == 768


@pytest.mark.asyncio
async def test_embed_raises_on_http_error(tmp_data_dir, monkeypatch):
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    import importlib
    import config
    importlib.reload(config)
    import embedder
    importlib.reload(embedder)
    with respx.mock:
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(500)
        )
        with pytest.raises(httpx.HTTPStatusError):
            await embedder.embed("hello")
