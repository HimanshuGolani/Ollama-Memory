# tests/test_routes.py
import pytest
import respx
import httpx
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_data_dir):
    import importlib
    import chroma_client, db
    import layers.code as lc, layers.notes as ln, layers.history as lh
    import routes.context, routes.index_routes, routes.remember, routes.status
    import main
    for mod in [chroma_client, db, lc, ln, lh,
                routes.context, routes.index_routes, routes.remember, routes.status, main]:
        importlib.reload(mod)
    from db import init_db
    init_db()
    from main import app
    return TestClient(app)


def test_status_ok(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "projects" in data


def test_context_code_empty_project(client):
    with respx.mock:
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": [0.1] * 768})
        )
        resp = client.post(
            "/context/code?project=/empty/proj",
            json={"query": "auth logic", "fullInput": "auth logic"},
        )
    assert resp.status_code == 200
    assert resp.json() == {"contents": []}


def test_remember_saves_note(client):
    with respx.mock:
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": [0.5] * 768})
        )
        resp = client.post(
            "/remember",
            json={"note": "Auth uses JWT", "project": "/my/proj"},
        )
    assert resp.status_code == 200
    assert resp.json()["saved"] is True


def test_context_notes_after_save(client):
    with respx.mock:
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": [0.8] * 768})
        )
        client.post("/remember", json={"note": "DB is Postgres on port 5432", "project": "/p"})
        resp = client.post(
            "/context/notes?project=/p",
            json={"query": "database", "fullInput": "database"},
        )
    data = resp.json()
    assert len(data["contents"]) >= 1
    assert "Postgres" in data["contents"][0]["content"]
