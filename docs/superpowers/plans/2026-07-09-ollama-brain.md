# Ollama Brain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local FastAPI memory server that gives Ollama models persistent knowledge of any codebase via three memory layers (RAG code index, curated notes, conversation history), surfaced in VS Code and IntelliJ IDEA through Continue.dev context providers.

**Architecture:** A Python FastAPI server runs on `localhost:11435`. ChromaDB stores all vector embeddings; SQLite stores structured metadata (notes, history, projects). Continue.dev's HTTP context provider calls the server on every chat query, injecting relevant code chunks, notes, and history summaries into the model's context window. Embeddings are generated via Ollama's `nomic-embed-text` model — no extra model downloads required.

**Tech Stack:** Python 3.11+ · FastAPI · Uvicorn · ChromaDB · SQLite3 · httpx · watchdog · typer · rich · pydantic-settings

## Global Constraints

- Python 3.11 minimum
- All data stored in `~/.ollama-brain/` (configurable via `OLLAMA_BRAIN_DATA_DIR` env var)
- Ollama must be running on `http://localhost:11434` (configurable via `OLLAMA_URL`)
- Server runs on port `11435` (configurable via `OLLAMA_BRAIN_PORT`)
- All context endpoints accept `POST` with body `{"query": "...", "fullInput": "..."}` and `?project=<absolute-path>` query param
- All context endpoints return `{"contents": [{"name": "...", "description": "...", "content": "..."}]}`
- No authentication, no remote calls beyond Ollama

---

## File Map

```
Ollama Brain/
├── config.py                  # Settings via pydantic-settings + env vars
├── db.py                      # SQLite connection, schema init, project helpers
├── embedder.py                # Async Ollama /api/embeddings wrapper
├── chroma_client.py           # ChromaDB persistent client + collection helpers
├── indexer.py                 # File discovery, language-aware chunking, index writes
├── watcher.py                 # watchdog FileSystemEventHandler, debounced re-index
├── layers/
│   ├── __init__.py            # empty
│   ├── code.py                # query ChromaDB code collection → format results
│   ├── notes.py               # query/write ChromaDB notes collection + SQLite
│   └── history.py             # store raw queries, summarize, query summaries
├── routes/
│   ├── __init__.py            # empty
│   ├── context.py             # POST /context/code|notes|history
│   ├── index_routes.py        # POST /index, DELETE /index
│   ├── remember.py            # POST /remember
│   └── status.py              # GET /status
├── main.py                    # FastAPI app, router registration, startup/shutdown
├── cli.py                     # typer CLI: index, notes, status, serve
├── requirements.txt
├── start.bat                  # Windows: start server in background
├── tests/
│   ├── conftest.py            # shared fixtures (tmp dirs, mock Ollama, test db)
│   ├── test_config.py
│   ├── test_db.py
│   ├── test_embedder.py
│   ├── test_chroma_client.py
│   ├── test_indexer.py
│   ├── test_layers.py
│   └── test_routes.py
└── docs/
    └── superpowers/
        ├── specs/             # design spec lives here
        └── plans/             # this file
```

---

## Task 1: Project scaffold, requirements, and config

**Files:**
- Create: `requirements.txt`
- Create: `config.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`
- Create: `layers/__init__.py`
- Create: `routes/__init__.py`

**Interfaces:**
- Produces: `settings` singleton (imported as `from config import settings`) with fields:
  - `settings.ollama_url: str` — `"http://localhost:11434"`
  - `settings.embed_model: str` — `"nomic-embed-text"`
  - `settings.chat_model: str` — `"llama3.2"` (the Ollama model used for summarization)
- `settings.server_port: int` — `11435`
  - `settings.data_dir: Path` — `~/.ollama-brain`

- [ ] **Step 1: Write requirements.txt**

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
chromadb==0.5.15
watchdog==4.0.2
httpx==0.27.2
rich==13.9.2
typer==0.12.5
pydantic-settings==2.5.2
pytest==8.3.3
pytest-asyncio==0.24.0
respx==0.21.1
```

- [ ] **Step 2: Install dependencies**

```bash
cd "Ollama Brain"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 3: Write test_config.py**

```python
# tests/test_config.py
import os
from pathlib import Path

def test_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("OLLAMA_BRAIN_PORT", raising=False)
    monkeypatch.delenv("OLLAMA_BRAIN_DATA_DIR", raising=False)
    import importlib, config
    importlib.reload(config)
    s = config.Settings()
    assert s.ollama_url == "http://localhost:11434"
    assert s.embed_model == "nomic-embed-text"
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
```

- [ ] **Step 4: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 5: Write config.py**

```python
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    chat_model: str = "llama3.2"
    server_port: int = 11435
    data_dir: Path = Path.home() / ".ollama-brain"

    model_config = {"env_prefix": "OLLAMA_BRAIN_", "env_file": ".env"}

settings = Settings()
```

- [ ] **Step 6: Create empty init files**

```python
# layers/__init__.py  — empty
# routes/__init__.py  — empty
```

- [ ] **Step 7: Write tests/conftest.py**

```python
import pytest
from pathlib import Path
import sqlite3

@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("config.settings.data_dir", tmp_path)
    return tmp_path
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: 2 PASSED

- [ ] **Step 9: Commit**

```bash
git init
git add requirements.txt config.py layers/__init__.py routes/__init__.py tests/
git commit -m "feat: project scaffold and settings config"
```

---

## Task 2: Database layer

**Files:**
- Create: `db.py`
- Create: `tests/test_db.py`

**Interfaces:**
- Consumes: `settings.data_dir`
- Produces:
  - `init_db()` — creates all tables, idempotent
  - `get_db() -> sqlite3.Connection` — returns open connection with `row_factory = sqlite3.Row`
  - `get_or_create_project(conn, path: str) -> int` — returns `projects.id`
  - `get_project_id(conn, path: str) -> int | None`

- [ ] **Step 1: Write test_db.py**

```python
# tests/test_db.py
import sqlite3
import pytest
from pathlib import Path

def test_init_db_creates_tables(tmp_data_dir):
    from db import init_db, get_db
    init_db()
    conn = get_db()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"projects", "notes", "history_raw", "history_summaries"} <= tables
    conn.close()

def test_init_db_is_idempotent(tmp_data_dir):
    from db import init_db
    init_db()
    init_db()  # should not raise

def test_get_or_create_project(tmp_data_dir):
    from db import init_db, get_db, get_or_create_project
    init_db()
    conn = get_db()
    pid1 = get_or_create_project(conn, "/my/project")
    pid2 = get_or_create_project(conn, "/my/project")
    assert pid1 == pid2
    assert isinstance(pid1, int)
    conn.close()

def test_get_project_id_missing(tmp_data_dir):
    from db import init_db, get_db, get_project_id
    init_db()
    conn = get_db()
    assert get_project_id(conn, "/does/not/exist") is None
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_db.py -v
```

Expected: `ModuleNotFoundError: No module named 'db'`

- [ ] **Step 3: Write db.py**

```python
import sqlite3
from pathlib import Path
from config import settings

def get_db() -> sqlite3.Connection:
    db_path = settings.data_dir / "memory.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db() -> None:
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            name TEXT,
            indexed_at DATETIME,
            config_json TEXT
        );
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            tags TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            chroma_id TEXT
        );
        CREATE TABLE IF NOT EXISTS history_raw (
            id INTEGER PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            query TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS history_summaries (
            id INTEGER PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            summary TEXT NOT NULL,
            chroma_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

def get_or_create_project(conn: sqlite3.Connection, path: str) -> int:
    row = conn.execute("SELECT id FROM projects WHERE path = ?", (path,)).fetchone()
    if row:
        return row["id"]
    cursor = conn.execute("INSERT INTO projects (path, name) VALUES (?, ?)",
                          (path, Path(path).name))
    conn.commit()
    return cursor.lastrowid

def get_project_id(conn: sqlite3.Connection, path: str) -> int | None:
    row = conn.execute("SELECT id FROM projects WHERE path = ?", (path,)).fetchone()
    return row["id"] if row else None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_db.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add db.py tests/test_db.py
git commit -m "feat: sqlite database layer with schema init"
```

---

## Task 3: Embedder and ChromaDB client

**Files:**
- Create: `embedder.py`
- Create: `chroma_client.py`
- Create: `tests/test_embedder.py`
- Create: `tests/test_chroma_client.py`

**Interfaces:**
- Consumes: `settings.ollama_url`, `settings.embed_model`, `settings.data_dir`
- Produces:
  - `async embed(text: str) -> list[float]` — calls Ollama, returns embedding vector
  - `get_collection(prefix: str, project_path: str) -> chromadb.Collection` — returns named collection
  - `project_hash(project_path: str) -> str` — 8-char hex hash of path

- [ ] **Step 1: Write test_embedder.py**

```python
# tests/test_embedder.py
import pytest
import respx
import httpx

@pytest.mark.asyncio
async def test_embed_returns_vector(tmp_data_dir):
    import importlib, embedder
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
async def test_embed_raises_on_http_error(tmp_data_dir):
    import importlib, embedder
    importlib.reload(embedder)
    with respx.mock:
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(500)
        )
        with pytest.raises(httpx.HTTPStatusError):
            await embedder.embed("hello")
```

- [ ] **Step 2: Write test_chroma_client.py**

```python
# tests/test_chroma_client.py
def test_project_hash_is_stable():
    from chroma_client import project_hash
    h1 = project_hash("/my/project")
    h2 = project_hash("/my/project")
    assert h1 == h2
    assert len(h1) == 8

def test_project_hash_differs_for_different_paths():
    from chroma_client import project_hash
    assert project_hash("/project/a") != project_hash("/project/b")

def test_get_collection_creates_and_reuses(tmp_data_dir):
    import importlib, chroma_client
    importlib.reload(chroma_client)
    col1 = chroma_client.get_collection("code", "/my/project")
    col2 = chroma_client.get_collection("code", "/my/project")
    assert col1.name == col2.name

def test_different_prefixes_give_different_collections(tmp_data_dir):
    import importlib, chroma_client
    importlib.reload(chroma_client)
    c1 = chroma_client.get_collection("code", "/my/project")
    c2 = chroma_client.get_collection("notes", "/my/project")
    assert c1.name != c2.name
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_embedder.py tests/test_chroma_client.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 4: Write embedder.py**

```python
import httpx
from config import settings

async def embed(text: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.ollama_url}/api/embeddings",
            json={"model": settings.embed_model, "prompt": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
```

- [ ] **Step 5: Write chroma_client.py**

```python
import hashlib
import chromadb
from config import settings

_client: chromadb.PersistentClient | None = None

def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        chroma_dir = settings.data_dir / "chroma"
        chroma_dir.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(chroma_dir))
    return _client

def project_hash(project_path: str) -> str:
    return hashlib.md5(project_path.encode()).hexdigest()[:8]

def get_collection(prefix: str, project_path: str) -> chromadb.Collection:
    name = f"{prefix}_{project_hash(project_path)}"
    return _get_client().get_or_create_collection(
        name,
        metadata={"hnsw:space": "cosine"},
    )
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_embedder.py tests/test_chroma_client.py -v
```

Expected: 6 PASSED

- [ ] **Step 7: Commit**

```bash
git add embedder.py chroma_client.py tests/test_embedder.py tests/test_chroma_client.py
git commit -m "feat: ollama embedder and chromadb client"
```

---

## Task 4: File chunker

**Files:**
- Create: `indexer.py` (chunking portion only)
- Create: `tests/test_indexer.py` (chunking tests only)

**Interfaces:**
- Produces:
  - `chunk_file(path: Path) -> list[dict]` — returns `[{"content": str, "start_line": int, "end_line": int}]`

- [ ] **Step 1: Write chunking tests**

```python
# tests/test_indexer.py
from pathlib import Path
import pytest

def test_chunk_python_file(tmp_path):
    from indexer import chunk_file
    f = tmp_path / "example.py"
    f.write_text(
        "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
    )
    chunks = chunk_file(f)
    assert len(chunks) >= 1
    assert all("content" in c and "start_line" in c and "end_line" in c for c in chunks)
    full = "\n".join(c["content"] for c in chunks)
    assert "def foo" in full
    assert "def bar" in full

def test_chunk_unknown_extension_uses_sliding_window(tmp_path):
    from indexer import chunk_file
    f = tmp_path / "notes.txt"
    f.write_text("line\n" * 400)
    chunks = chunk_file(f)
    assert len(chunks) >= 2

def test_chunk_empty_file_returns_empty(tmp_path):
    from indexer import chunk_file
    f = tmp_path / "empty.py"
    f.write_text("")
    assert chunk_file(f) == []

def test_chunk_preserves_line_numbers(tmp_path):
    from indexer import chunk_file
    f = tmp_path / "sample.py"
    f.write_text("a = 1\nb = 2\ndef foo():\n    pass\n")
    chunks = chunk_file(f)
    assert chunks[0]["start_line"] == 1
    for c in chunks:
        assert c["start_line"] <= c["end_line"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_indexer.py -v
```

Expected: `ModuleNotFoundError: No module named 'indexer'`

- [ ] **Step 3: Write indexer.py (chunking functions only)**

```python
import re
from pathlib import Path

LANGUAGE_PATTERNS: dict[str, str] = {
    ".py":   r"^(def |class |\s{0,4}def |\s{0,4}class )",
    ".js":   r"^(function |const |class |export )",
    ".ts":   r"^(function |const |class |export |interface |type )",
    ".java": r"^\s*(public|private|protected|static|class|interface|enum)\s",
    ".kt":   r"^\s*(fun |class |object |interface )",
    ".go":   r"^func ",
    ".cs":   r"^\s*(public|private|protected|static|class|interface|namespace)\s",
}

WINDOW_CHARS = 1200
OVERLAP_LINES = 3

SKIP_DIRS = {
    "node_modules", "__pycache__", ".git", ".venv", "venv",
    "dist", "build", ".idea", ".vscode", "target", "bin", "obj",
}

def chunk_file(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    if not text.strip():
        return []
    pattern = LANGUAGE_PATTERNS.get(path.suffix.lower())
    raw = _split_by_pattern(text, pattern) if pattern else _sliding_window(text)
    return [c for c in raw if c["content"].strip()]

def _split_by_pattern(text: str, pattern: str) -> list[dict]:
    lines = text.split("\n")
    chunks: list[dict] = []
    current: list[str] = []
    start = 0
    for i, line in enumerate(lines):
        if re.match(pattern, line) and current:
            chunks.append(_make_chunk(current, start))
            current = [line]
            start = i
        else:
            current.append(line)
    if current:
        chunks.append(_make_chunk(current, start))
    return _merge_tiny(chunks)

def _sliding_window(text: str) -> list[dict]:
    lines = text.split("\n")
    chunks: list[dict] = []
    current: list[str] = []
    chars = 0
    start = 0
    for i, line in enumerate(lines):
        current.append(line)
        chars += len(line) + 1
        if chars >= WINDOW_CHARS:
            chunks.append(_make_chunk(current, start))
            overlap = current[-OVERLAP_LINES:]
            current = list(overlap)
            chars = sum(len(l) + 1 for l in current)
            start = i - len(overlap) + 1
    if current:
        chunks.append(_make_chunk(current, start))
    return chunks

def _make_chunk(lines: list[str], start_idx: int) -> dict:
    return {
        "content": "\n".join(lines),
        "start_line": start_idx + 1,
        "end_line": start_idx + len(lines),
    }

def _merge_tiny(chunks: list[dict], min_chars: int = 50) -> list[dict]:
    merged: list[dict] = []
    for chunk in chunks:
        if merged and len(merged[-1]["content"]) < min_chars:
            merged[-1]["content"] += "\n" + chunk["content"]
            merged[-1]["end_line"] = chunk["end_line"]
        else:
            merged.append(chunk)
    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_indexer.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add indexer.py tests/test_indexer.py
git commit -m "feat: language-aware file chunker"
```

---

## Task 5: Indexer orchestration and file watcher

**Files:**
- Modify: `indexer.py` (add indexing functions)
- Create: `watcher.py`
- Modify: `tests/test_indexer.py` (add indexing tests)

**Interfaces:**
- Consumes: `chunk_file`, `embed`, `get_collection`, `get_db`, `get_or_create_project`, `SKIP_DIRS`
- Produces:
  - `async index_project(project_path: str) -> int` — indexes all files, returns chunk count
  - `async index_file(project_path: str, file_path: Path) -> int` — re-indexes one file
  - `clear_project_index(project_path: str) -> None` — removes all vectors for project
  - `start_watcher(project_path: str) -> Observer` — returns started watchdog Observer
  - `stop_watcher(observer: Observer) -> None`

- [ ] **Step 1: Add indexing tests to test_indexer.py**

```python
# append to tests/test_indexer.py
import pytest

@pytest.mark.asyncio
async def test_index_project(tmp_path, tmp_data_dir):
    import respx, httpx, importlib
    import chroma_client, indexer
    importlib.reload(chroma_client)
    importlib.reload(indexer)

    proj = tmp_path / "myproject"
    proj.mkdir()
    (proj / "foo.py").write_text("def hello():\n    return 'hi'\n")

    with respx.mock:
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": [0.1] * 768})
        )
        from indexer import index_project
        count = await index_project(str(proj))

    assert count >= 1

@pytest.mark.asyncio
async def test_clear_project_index(tmp_path, tmp_data_dir):
    import respx, httpx, importlib
    import chroma_client, indexer
    importlib.reload(chroma_client)
    importlib.reload(indexer)

    proj = tmp_path / "myproject2"
    proj.mkdir()
    (proj / "bar.py").write_text("def bar():\n    pass\n")

    with respx.mock:
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": [0.1] * 768})
        )
        from indexer import index_project, clear_project_index
        await index_project(str(proj))
        clear_project_index(str(proj))

    col = chroma_client.get_collection("code", str(proj))
    assert col.count() == 0
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
pytest tests/test_indexer.py::test_index_project tests/test_indexer.py::test_clear_project_index -v
```

Expected: `ImportError` (functions not yet defined)

- [ ] **Step 3: Add indexing functions to indexer.py**

Append to the bottom of `indexer.py`:

```python
import asyncio
import json
from datetime import datetime, timezone
from db import get_db, get_or_create_project, init_db
from embedder import embed
from chroma_client import get_collection

SOURCE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".kt", ".go",
    ".cs", ".cpp", ".c", ".h", ".rs", ".rb", ".php", ".swift",
    ".md", ".txt", ".yaml", ".yml", ".json", ".toml",
}

def _should_index(path: Path) -> bool:
    for part in path.parts:
        if part in SKIP_DIRS:
            return False
    return path.suffix.lower() in SOURCE_EXTENSIONS

def _discover_files(project_path: str) -> list[Path]:
    root = Path(project_path)
    return [p for p in root.rglob("*") if p.is_file() and _should_index(p)]

async def index_project(project_path: str) -> int:
    init_db()
    conn = get_db()
    pid = get_or_create_project(conn, project_path)
    conn.execute("UPDATE projects SET indexed_at = ? WHERE id = ?",
                 (datetime.now(timezone.utc).isoformat(), pid))
    conn.commit()
    conn.close()

    files = _discover_files(project_path)
    total = 0
    for file_path in files:
        total += await index_file(project_path, file_path)
    return total

async def index_file(project_path: str, file_path: Path) -> int:
    chunks = chunk_file(file_path)
    if not chunks:
        return 0
    collection = get_collection("code", project_path)
    rel = str(file_path.relative_to(project_path))

    # Remove stale chunks for this file
    try:
        existing = collection.get(where={"file": rel})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    ids, embeddings, documents, metadatas = [], [], [], []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{rel}::{chunk['start_line']}"
        vec = await embed(chunk["content"])
        ids.append(chunk_id)
        embeddings.append(vec)
        documents.append(chunk["content"])
        metadatas.append({
            "file": rel,
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
        })

    collection.upsert(ids=ids, embeddings=embeddings,
                      documents=documents, metadatas=metadatas)
    return len(chunks)

def clear_project_index(project_path: str) -> None:
    collection = get_collection("code", project_path)
    all_ids = collection.get()["ids"]
    if all_ids:
        collection.delete(ids=all_ids)
```

- [ ] **Step 4: Write watcher.py**

```python
import asyncio
import time
from pathlib import Path
from threading import Thread
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent

class _IndexHandler(FileSystemEventHandler):
    def __init__(self, project_path: str, loop: asyncio.AbstractEventLoop):
        self._project = project_path
        self._loop = loop
        self._pending: dict[str, float] = {}
        self._debounce = 2.0

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def _schedule(self, path: str):
        self._pending[path] = time.monotonic() + self._debounce
        Thread(target=self._flush, args=(path,), daemon=True).start()

    def _flush(self, path: str):
        time.sleep(self._debounce + 0.1)
        if path not in self._pending:
            return
        due = self._pending.pop(path, 0)
        if time.monotonic() < due:
            return
        from indexer import index_file
        asyncio.run_coroutine_threadsafe(
            index_file(self._project, Path(path)),
            self._loop,
        )

def start_watcher(project_path: str) -> Observer:
    loop = asyncio.get_event_loop()
    handler = _IndexHandler(project_path, loop)
    observer = Observer()
    observer.schedule(handler, project_path, recursive=True)
    observer.start()
    return observer

def stop_watcher(observer: Observer) -> None:
    observer.stop()
    observer.join()
```

- [ ] **Step 5: Run all indexer tests**

```bash
pytest tests/test_indexer.py -v
```

Expected: 6 PASSED

- [ ] **Step 6: Commit**

```bash
git add indexer.py watcher.py tests/test_indexer.py
git commit -m "feat: project indexer with file watcher"
```

---

## Task 6: Code memory layer

**Files:**
- Create: `layers/code.py`
- Create: `tests/test_layers.py`

**Interfaces:**
- Consumes: `embed`, `get_collection`
- Produces:
  - `async query_code(query: str, project_path: str, n_results: int = 8) -> list[dict]`
    — returns `[{"name": "file.py:42", "description": "lines 42-60", "content": "..."}]`

- [ ] **Step 1: Write code layer tests**

```python
# tests/test_layers.py
import pytest
import respx
import httpx
import importlib

@pytest.fixture(autouse=True)
def reload_modules(tmp_data_dir):
    import chroma_client, layers.code as lc
    importlib.reload(chroma_client)
    importlib.reload(lc)

@pytest.mark.asyncio
async def test_query_code_returns_results(tmp_data_dir):
    import chroma_client
    from layers.code import query_code

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
        results = await query_code("hello function", "/my/proj", n_results=1)

    assert len(results) == 1
    assert results[0]["name"] == "foo.py:1"
    assert "hello" in results[0]["content"]

@pytest.mark.asyncio
async def test_query_code_empty_index(tmp_data_dir):
    from layers.code import query_code
    with respx.mock:
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": [0.1] * 768})
        )
        results = await query_code("anything", "/empty/project", n_results=5)
    assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_layers.py::test_query_code_returns_results tests/test_layers.py::test_query_code_empty_index -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write layers/code.py**

```python
from embedder import embed
from chroma_client import get_collection

async def query_code(query: str, project_path: str, n_results: int = 8) -> list[dict]:
    collection = get_collection("code", project_path)
    if collection.count() == 0:
        return []
    vec = await embed(query)
    results = collection.query(
        query_embeddings=[vec],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas"],
    )
    output = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        file_name = meta.get("file", "unknown")
        start = meta.get("start_line", 0)
        end = meta.get("end_line", 0)
        output.append({
            "name": f"{file_name}:{start}",
            "description": f"lines {start}–{end}",
            "content": doc,
        })
    return output
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_layers.py::test_query_code_returns_results tests/test_layers.py::test_query_code_empty_index -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add layers/code.py tests/test_layers.py
git commit -m "feat: code memory layer with semantic search"
```

---

## Task 7: Notes memory layer

**Files:**
- Create: `layers/notes.py`
- Modify: `tests/test_layers.py` (add notes tests)

**Interfaces:**
- Consumes: `embed`, `get_collection`, `get_db`, `get_or_create_project`, `init_db`
- Produces:
  - `async save_note(note: str, project_path: str) -> int` — saves to SQLite + ChromaDB, returns note id
  - `async query_notes(query: str, project_path: str, n_results: int = 5) -> list[dict]`
    — returns `[{"name": "Note #3", "description": "saved 2026-07-09", "content": "..."}]`

- [ ] **Step 1: Append notes tests to test_layers.py**

```python
# append to tests/test_layers.py
@pytest.mark.asyncio
async def test_save_and_query_note(tmp_data_dir):
    import importlib, chroma_client, layers.notes as ln
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

@pytest.mark.asyncio
async def test_query_notes_empty(tmp_data_dir):
    import importlib, chroma_client, layers.notes as ln
    importlib.reload(chroma_client)
    importlib.reload(ln)
    from db import init_db
    init_db()

    with respx.mock:
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": [0.1] * 768})
        )
        results = await ln.query_notes("nothing here", "/empty/proj")
    assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_layers.py::test_save_and_query_note tests/test_layers.py::test_query_notes_empty -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write layers/notes.py**

```python
import uuid
from datetime import datetime, timezone
from db import get_db, get_or_create_project, init_db
from embedder import embed
from chroma_client import get_collection

async def save_note(note: str, project_path: str) -> int:
    init_db()
    vec = await embed(note)
    chroma_id = str(uuid.uuid4())

    collection = get_collection("notes", project_path)
    collection.upsert(
        ids=[chroma_id],
        embeddings=[vec],
        documents=[note],
        metadatas=[{"project": project_path}],
    )

    conn = get_db()
    pid = get_or_create_project(conn, project_path)
    cursor = conn.execute(
        "INSERT INTO notes (project_id, content, chroma_id) VALUES (?, ?, ?)",
        (pid, note, chroma_id),
    )
    conn.commit()
    note_id = cursor.lastrowid
    conn.close()
    return note_id

async def query_notes(query: str, project_path: str, n_results: int = 5) -> list[dict]:
    collection = get_collection("notes", project_path)
    if collection.count() == 0:
        return []
    vec = await embed(query)
    results = collection.query(
        query_embeddings=[vec],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas"],
    )
    conn = get_db()
    output = []
    for chroma_id, doc in zip(results["ids"][0], results["documents"][0]):
        row = conn.execute(
            "SELECT id, created_at FROM notes WHERE chroma_id = ?", (chroma_id,)
        ).fetchone()
        saved = row["created_at"][:10] if row else "unknown"
        note_num = row["id"] if row else "?"
        output.append({
            "name": f"Note #{note_num}",
            "description": f"saved {saved}",
            "content": doc,
        })
    conn.close()
    return output
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_layers.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add layers/notes.py tests/test_layers.py
git commit -m "feat: notes memory layer with save and semantic search"
```

---

## Task 8: History memory layer

**Files:**
- Create: `layers/history.py`
- Modify: `tests/test_layers.py` (add history tests)

**Interfaces:**
- Consumes: `embed`, `get_collection`, `get_db`, `get_or_create_project`, `init_db`, `settings.ollama_url`
- Produces:
  - `async record_query(query: str, project_path: str) -> None` — stores raw query, triggers summarization every 10 queries
  - `async query_history(query: str, project_path: str, n_results: int = 3) -> list[dict]`
    — returns `[{"name": "Past Q&A", "description": "from 2026-07-09", "content": "..."}]`
  - `async _summarize_batch(project_id: int, project_path: str, queries: list[str]) -> None` — internal, calls Ollama chat

- [ ] **Step 1: Append history tests to test_layers.py**

```python
# append to tests/test_layers.py
@pytest.mark.asyncio
async def test_record_and_query_history(tmp_data_dir):
    import importlib, chroma_client
    import layers.history as lh
    importlib.reload(chroma_client)
    importlib.reload(lh)
    from db import init_db
    init_db()

    # Directly insert a summary (bypasses Ollama summarization)
    import uuid
    col = chroma_client.get_collection("history", "/my/proj")
    cid = str(uuid.uuid4())
    col.add(
        ids=[cid],
        embeddings=[[0.7] * 768],
        documents=["Q: How does auth work? A: It uses JWT tokens stored in .env"],
        metadatas=[{"project": "/my/proj"}],
    )
    from db import get_db, get_or_create_project
    conn = get_db()
    pid = get_or_create_project(conn, "/my/proj")
    conn.execute(
        "INSERT INTO history_summaries (project_id, summary, chroma_id) VALUES (?,?,?)",
        (pid, "Q: How does auth work? A: JWT tokens", cid)
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

@pytest.mark.asyncio
async def test_query_history_empty(tmp_data_dir):
    import importlib, chroma_client
    import layers.history as lh
    importlib.reload(chroma_client)
    importlib.reload(lh)
    from db import init_db
    init_db()

    with respx.mock:
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": [0.1] * 768})
        )
        results = await lh.query_history("nothing", "/empty/proj")
    assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_layers.py::test_record_and_query_history tests/test_layers.py::test_query_history_empty -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write layers/history.py**

```python
import uuid
import httpx
from datetime import datetime, timezone
from db import get_db, get_or_create_project, init_db
from embedder import embed
from chroma_client import get_collection
from config import settings

SUMMARIZE_EVERY = 10

async def record_query(query: str, project_path: str) -> None:
    init_db()
    conn = get_db()
    pid = get_or_create_project(conn, project_path)
    conn.execute("INSERT INTO history_raw (project_id, query) VALUES (?, ?)", (pid, query))
    conn.commit()
    count = conn.execute(
        "SELECT COUNT(*) as c FROM history_raw WHERE project_id = ?", (pid,)
    ).fetchone()["c"]
    if count % SUMMARIZE_EVERY == 0:
        rows = conn.execute(
            "SELECT query FROM history_raw WHERE project_id = ? ORDER BY created_at DESC LIMIT ?",
            (pid, SUMMARIZE_EVERY),
        ).fetchall()
        queries = [r["query"] for r in rows]
        conn.close()
        await _summarize_batch(pid, project_path, queries)
    else:
        conn.close()

async def _summarize_batch(project_id: int, project_path: str, queries: list[str]) -> None:
    prompt = (
        "Summarize these coding assistant queries into 2-3 concise bullet points "
        "capturing the key topics and answers discussed. Be specific about code and files mentioned.\n\n"
        + "\n".join(f"- {q}" for q in queries)
    )
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{settings.ollama_url}/api/generate",
                json={"model": settings.chat_model, "prompt": prompt, "stream": False},
                timeout=60.0,
            )
            resp.raise_for_status()
            summary = resp.json().get("response", "")
    except Exception:
        summary = "; ".join(queries[:3])

    if not summary.strip():
        return

    vec = await embed(summary)
    chroma_id = str(uuid.uuid4())
    collection = get_collection("history", project_path)
    collection.upsert(
        ids=[chroma_id],
        embeddings=[vec],
        documents=[summary],
        metadatas=[{"project": project_path}],
    )
    conn = get_db()
    conn.execute(
        "INSERT INTO history_summaries (project_id, summary, chroma_id) VALUES (?, ?, ?)",
        (project_id, summary, chroma_id),
    )
    conn.commit()
    conn.close()

async def query_history(query: str, project_path: str, n_results: int = 3) -> list[dict]:
    collection = get_collection("history", project_path)
    if collection.count() == 0:
        return []
    vec = await embed(query)
    results = collection.query(
        query_embeddings=[vec],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas"],
    )
    conn = get_db()
    output = []
    for chroma_id, doc in zip(results["ids"][0], results["documents"][0]):
        row = conn.execute(
            "SELECT created_at FROM history_summaries WHERE chroma_id = ?", (chroma_id,)
        ).fetchone()
        date = row["created_at"][:10] if row else "unknown"
        output.append({
            "name": "Past Q&A",
            "description": f"from {date}",
            "content": doc,
        })
    conn.close()
    return output
```

- [ ] **Step 4: Run all layer tests**

```bash
pytest tests/test_layers.py -v
```

Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add layers/history.py tests/test_layers.py
git commit -m "feat: conversation history layer with auto-summarization"
```

---

## Task 9: FastAPI routes and server

**Files:**
- Create: `routes/context.py`
- Create: `routes/index_routes.py`
- Create: `routes/remember.py`
- Create: `routes/status.py`
- Create: `main.py`
- Create: `tests/test_routes.py`

**Interfaces:**
- Consumes: `query_code`, `query_notes`, `query_history`, `record_query`, `save_note`, `index_project`, `clear_project_index`, `start_watcher`, `stop_watcher`, `get_db`, `init_db`
- Produces: Running FastAPI app on `settings.server_port`

- [ ] **Step 1: Write test_routes.py**

```python
# tests/test_routes.py
import pytest
import respx
import httpx
from fastapi.testclient import TestClient

@pytest.fixture
def client(tmp_data_dir):
    import importlib
    import chroma_client, db, layers.code as lc, layers.notes as ln, layers.history as lh
    import routes.context, routes.index_routes, routes.remember, routes.status, main
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
    assert "status" in data
    assert data["status"] == "ok"

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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_routes.py -v
```

Expected: `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Write routes/context.py**

```python
from fastapi import APIRouter, Query
from pydantic import BaseModel
from layers.code import query_code
from layers.notes import query_notes
from layers.history import query_history, record_query

router = APIRouter()

class QueryBody(BaseModel):
    query: str
    fullInput: str = ""

@router.post("/context/code")
async def context_code(body: QueryBody, project: str = Query(...)):
    await record_query(body.query, project)
    results = await query_code(body.query, project)
    return {"contents": results}

@router.post("/context/notes")
async def context_notes(body: QueryBody, project: str = Query(...)):
    results = await query_notes(body.query, project)
    return {"contents": results}

@router.post("/context/history")
async def context_history(body: QueryBody, project: str = Query(...)):
    results = await query_history(body.query, project)
    return {"contents": results}
```

- [ ] **Step 4: Write routes/index_routes.py**

```python
from fastapi import APIRouter
from pydantic import BaseModel
from indexer import index_project, clear_project_index

router = APIRouter()

class ProjectBody(BaseModel):
    project: str

@router.post("/index")
async def trigger_index(body: ProjectBody):
    count = await index_project(body.project)
    return {"indexed_chunks": count, "project": body.project}

@router.delete("/index")
async def delete_index(body: ProjectBody):
    clear_project_index(body.project)
    return {"cleared": True, "project": body.project}
```

- [ ] **Step 5: Write routes/remember.py**

```python
from fastapi import APIRouter
from pydantic import BaseModel
from layers.notes import save_note

router = APIRouter()

class NoteBody(BaseModel):
    note: str
    project: str

@router.post("/remember")
async def remember(body: NoteBody):
    note_id = await save_note(body.note, body.project)
    return {"saved": True, "id": note_id}
```

- [ ] **Step 6: Write routes/status.py**

```python
from fastapi import APIRouter
from db import get_db, init_db

router = APIRouter()

@router.get("/status")
def status():
    init_db()
    conn = get_db()
    projects = conn.execute("SELECT path, indexed_at FROM projects").fetchall()
    result = [{"path": r["path"], "indexed_at": r["indexed_at"]} for r in projects]
    conn.close()
    return {"status": "ok", "projects": result}
```

- [ ] **Step 7: Write main.py**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from db import init_db
from routes.context import router as context_router
from routes.index_routes import router as index_router
from routes.remember import router as remember_router
from routes.status import router as status_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Ollama Brain", lifespan=lifespan)
app.include_router(context_router)
app.include_router(index_router)
app.include_router(remember_router)
app.include_router(status_router)
```

- [ ] **Step 8: Run all route tests**

```bash
pytest tests/test_routes.py -v
```

Expected: 4 PASSED

- [ ] **Step 9: Run entire test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass

- [ ] **Step 10: Commit**

```bash
git add routes/ main.py tests/test_routes.py
git commit -m "feat: fastapi routes and server entry point"
```

---

## Task 10: CLI, startup scripts, and Continue.dev config

**Files:**
- Create: `cli.py`
- Create: `start.bat`
- Create: `configure_continue.py`

**Interfaces:**
- Consumes: `index_project`, `save_note`, `get_db`, `init_db`, `settings`
- Produces: runnable `python cli.py` commands + startup bat + `continue_config_snippet.json` generator

- [ ] **Step 1: Write cli.py**

```python
import asyncio
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Ollama Brain — persistent memory for your codebase")
console = Console()

@app.command()
def serve():
    """Start the memory server."""
    import uvicorn
    from config import settings
    console.print(f"[green]Starting Ollama Brain on port {settings.server_port}[/green]")
    uvicorn.run("main:app", host="0.0.0.0", port=settings.server_port, reload=False)

@app.command()
def index(project: str = typer.Argument(..., help="Absolute path to project root")):
    """Index a project's source files into memory."""
    from db import init_db
    init_db()
    console.print(f"[cyan]Indexing {project}...[/cyan]")
    from indexer import index_project
    count = asyncio.run(index_project(project))
    console.print(f"[green]Done. Indexed {count} chunks.[/green]")

@app.command()
def remember(
    note: str = typer.Argument(..., help="Note text to save"),
    project: str = typer.Option(..., help="Absolute path to project root"),
):
    """Save a note about a project."""
    from db import init_db
    init_db()
    from layers.notes import save_note
    note_id = asyncio.run(save_note(note, project))
    console.print(f"[green]Note #{note_id} saved.[/green]")

@app.command()
def status():
    """Show indexed projects and stats."""
    from db import init_db, get_db
    init_db()
    conn = get_db()
    rows = conn.execute("SELECT path, name, indexed_at FROM projects").fetchall()
    conn.close()
    if not rows:
        console.print("[yellow]No projects indexed yet.[/yellow]")
        return
    table = Table("Project", "Name", "Last Indexed")
    for r in rows:
        table.add_row(r["path"], r["name"] or "-", r["indexed_at"] or "never")
    console.print(table)

@app.command()
def configure(project: str = typer.Argument(..., help="Absolute path to your project")):
    """Print the Continue.dev config snippet for this project."""
    import json
    from config import settings
    snippet = {
        "contextProviders": [
            {
                "name": "http",
                "params": {
                    "url": f"http://localhost:{settings.server_port}/context/code?project={project}",
                    "title": "Code Memory",
                    "description": "Relevant code chunks from the indexed codebase",
                    "displayTitle": "code",
                },
            },
            {
                "name": "http",
                "params": {
                    "url": f"http://localhost:{settings.server_port}/context/notes?project={project}",
                    "title": "Project Notes",
                    "description": "Curated architecture notes and decisions",
                    "displayTitle": "notes",
                },
            },
            {
                "name": "http",
                "params": {
                    "url": f"http://localhost:{settings.server_port}/context/history?project={project}",
                    "title": "Conversation History",
                    "description": "Relevant past Q&A about this project",
                    "displayTitle": "history",
                },
            },
        ],
        "slashCommands": [
            {
                "name": "remember",
                "description": "Save a note about this codebase",
                "step": "HttpSlashCommand",
                "params": {"url": f"http://localhost:{settings.server_port}/remember"},
            }
        ],
    }
    console.print("\n[bold]Add this to ~/.continue/config.json:[/bold]\n")
    console.print(json.dumps(snippet, indent=2))

if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Write start.bat**

```bat
@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)
echo Starting Ollama Brain server...
python cli.py serve
```

- [ ] **Step 3: Smoke test the CLI**

```bash
python cli.py --help
```

Expected: lists `serve`, `index`, `remember`, `status`, `configure` commands.

```bash
python cli.py status
```

Expected: "No projects indexed yet." (or a table if you've indexed before)

- [ ] **Step 4: Test configure command**

```bash
python cli.py configure C:/Users/YourName/projects/my-project
```

Expected: prints a JSON snippet with three context providers using the given project path.

- [ ] **Step 5: Run full test suite one final time**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests pass

- [ ] **Step 6: Final commit**

```bash
git add cli.py start.bat
git commit -m "feat: cli commands and windows startup script"
```

---

## Usage Quickstart (after all tasks complete)

```bash
# 1. Start the server (keep this terminal open, or run start.bat)
python cli.py serve

# 2. Index your project (run once, then auto-updates on file save)
python cli.py index C:/Users/You/projects/my-project

# 3. Get your Continue.dev config snippet
python cli.py configure C:/Users/You/projects/my-project
# → copy the printed JSON into ~/.continue/config.json

# 4. In Continue.dev chat, type @code, @notes, or @history before your question
#    Example: @code how does the login flow work?

# 5. Save notes from Continue.dev chat:
#    /remember The payment service uses Stripe webhooks for refunds
```
