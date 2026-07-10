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


@pytest.mark.asyncio
async def test_save_and_query_note(tmp_data_dir):
    import chroma_client
    import layers.notes as ln
    importlib.reload(chroma_client)
    importlib.reload(ln)
    from db import init_db
    init_db()

    with respx.mock:
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": [0.8] * 768})
        )
        note_id = await ln.save_note("Auth uses JWT, secrets in .env", "/my/proj")
        results = await ln.query_notes("authentication", "/my/proj", n_results=1)

    assert isinstance(note_id, int)
    assert len(results) == 1
    assert "JWT" in results[0]["content"]
    assert results[0]["name"].startswith("Note #")
    assert "saved" in results[0]["description"]


@pytest.mark.asyncio
async def test_query_notes_empty(tmp_data_dir):
    import chroma_client
    import layers.notes as ln
    importlib.reload(chroma_client)
    importlib.reload(ln)
    from db import init_db
    init_db()

    with respx.mock:
        # embed should NOT be called for empty collection
        results = await ln.query_notes("nothing here", "/empty/proj")
    assert results == []


@pytest.mark.asyncio
async def test_record_and_query_history(tmp_data_dir):
    import importlib, chroma_client
    import layers.history as lh
    importlib.reload(chroma_client)
    importlib.reload(lh)
    from db import init_db, get_db, get_or_create_project
    init_db()

    # Directly insert a summary (bypasses Ollama summarization)
    import uuid
    col = chroma_client.get_collection("history", "/my/proj")
    cid = str(uuid.uuid4())
    col.add(
        ids=[cid],
        embeddings=[[0.7] * 768],
        documents=["Q: How does auth work? A: JWT tokens stored in .env"],
        metadatas=[{"project": "/my/proj"}],
    )
    conn = get_db()
    pid = get_or_create_project(conn, "/my/proj")
    conn.execute(
        "INSERT INTO history_summaries (project_id, summary, chroma_id) VALUES (?,?,?)",
        (pid, "Q: How does auth work? A: JWT tokens", cid),
    )
    conn.commit()
    conn.close()

    with respx.mock:
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": [0.7] * 768})
        )
        results = await lh.query_history("authentication", "/my/proj", n_results=1)

    assert len(results) == 1
    assert "JWT" in results[0]["content"]
    assert results[0]["name"] == "Past Q&A"
    assert "from" in results[0]["description"]


@pytest.mark.asyncio
async def test_query_history_empty(tmp_data_dir):
    import importlib, chroma_client
    import layers.history as lh
    importlib.reload(chroma_client)
    importlib.reload(lh)
    from db import init_db
    init_db()

    with respx.mock:
        # embed should NOT be called for empty collection
        results = await lh.query_history("nothing", "/empty/proj")
    assert results == []
