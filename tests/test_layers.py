import pytest
import respx
import httpx
import importlib


@pytest.fixture(autouse=True)
def reload_modules(tmp_data_dir):
    import chroma_client
    importlib.reload(chroma_client)


@pytest.mark.asyncio
async def test_query_code_returns_results(tmp_data_dir):
    import chroma_client
    import layers.code as lc
    importlib.reload(lc)

    col = chroma_client.get_collection("code", "/my/proj")
    col.add(
        ids=["foo.py::1"],
        embeddings=[[0.9] * 768],
        documents=["def hello(): return 'hi'"],
        metadatas=[{"file": "foo.py", "start_line": 1, "end_line": 3}],
    )

    with respx.mock:
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": [0.9] * 768})
        )
        results = await lc.query_code("hello function", "/my/proj", n_results=1)

    assert len(results) == 1
    assert results[0]["name"] == "foo.py:1"
    assert "hello" in results[0]["content"]
    assert "lines 1" in results[0]["description"]


@pytest.mark.asyncio
async def test_query_code_empty_index(tmp_data_dir):
    import layers.code as lc
    importlib.reload(lc)

    with respx.mock:
        # embed should NOT be called if collection is empty
        results = await lc.query_code("anything", "/empty/project", n_results=5)
    assert results == []
