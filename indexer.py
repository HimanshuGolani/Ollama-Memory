import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path

from chroma_client import get_collection
from db import get_db, get_or_create_project, init_db
from embedder import embed

LANGUAGE_PATTERNS: dict[str, str] = {
    ".py":   r"^(def |class |\s{0,4}def |\s{0,4}class )",
    ".js":   r"^(function |const |class |export )",
    ".ts":   r"^(function |const |class |export |interface |type )",
    ".java": r"^\s*(public|private|protected|static|class|interface|enum)\s",
    ".kt":   r"^\s*(fun |class |object |interface )",
    ".go":   r"^func ",
    ".cs":   r"^\s*(public|private|protected|static|class|interface|namespace)\s",
}

SKIP_DIRS: set[str] = {
    "node_modules", "__pycache__", ".git", ".venv", "venv",
    "dist", "build", ".idea", ".vscode", "target", "bin", "obj",
}

SOURCE_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".kt", ".go",
    ".cs", ".cpp", ".c", ".h", ".rs", ".rb", ".php", ".swift",
    ".md", ".txt", ".yaml", ".yml", ".json", ".toml",
}

WINDOW_CHARS = 1200
OVERLAP_LINES = 3


def chunk_file(path: Path) -> list[dict]:
    """
    Split a file into chunks, using language-specific patterns if available,
    otherwise using a sliding window approach.

    Returns:
        List of dicts with keys: "content", "start_line", "end_line"
        Empty list if file is empty or unreadable.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    if not text.strip():
        return []

    pattern = LANGUAGE_PATTERNS.get(path.suffix.lower())
    raw = _split_by_pattern(text, pattern) if pattern else _sliding_window(text)

    # Filter out chunks with empty content after stripping
    return [c for c in raw if c["content"].strip()]


def _split_by_pattern(text: str, pattern: str) -> list[dict]:
    """
    Split text by language-specific pattern (functions/classes).
    When a line matches the pattern and we have accumulated lines,
    start a new chunk.
    """
    lines = text.split("\n")
    chunks: list[dict] = []
    current: list[str] = []
    start = 0

    for i, line in enumerate(lines):
        if re.match(pattern, line) and current:
            # Pattern matched and we have content: save current chunk, start new one
            chunks.append(_make_chunk(current, start))
            current = [line]
            start = i
        else:
            # Just accumulate the line
            current.append(line)

    # Don't forget the last chunk
    if current:
        chunks.append(_make_chunk(current, start))

    return _merge_tiny(chunks)


def _sliding_window(text: str) -> list[dict]:
    """
    Split text using a sliding window approach: accumulate lines until
    we reach WINDOW_CHARS characters, then create a chunk with OVERLAP_LINES overlap.
    """
    lines = text.split("\n")
    chunks: list[dict] = []
    current: list[str] = []
    chars = 0
    start = 0

    for i, line in enumerate(lines):
        current.append(line)
        chars += len(line) + 1  # +1 for newline

        if chars >= WINDOW_CHARS:
            # Window full: save chunk
            chunks.append(_make_chunk(current, start))

            # Prepare overlap
            overlap = current[-OVERLAP_LINES:]
            current = list(overlap)
            chars = sum(len(l) + 1 for l in current)
            start = i - len(overlap) + 1

    # Don't forget the final chunk
    if current:
        chunks.append(_make_chunk(current, start))

    return chunks


def _make_chunk(lines: list[str], start_idx: int) -> dict:
    """Create a chunk dict from lines and starting index."""
    return {
        "content": "\n".join(lines),
        "start_line": start_idx + 1,  # 1-based
        "end_line": start_idx + len(lines),
    }


def _merge_tiny(chunks: list[dict], min_chars: int = 50) -> list[dict]:
    """
    Merge chunks shorter than min_chars into the previous chunk.
    """
    merged: list[dict] = []
    for chunk in chunks:
        if merged and len(merged[-1]["content"]) < min_chars:
            # Previous chunk is too small: merge this one into it
            merged[-1]["content"] += "\n" + chunk["content"]
            merged[-1]["end_line"] = chunk["end_line"]
        else:
            # Either no previous chunk or it's large enough: add as new chunk
            merged.append(chunk)
    return merged


# ---------------------------------------------------------------------------
# Indexing orchestration
# ---------------------------------------------------------------------------

def _should_index(path: Path) -> bool:
    for part in path.parts:
        if part in SKIP_DIRS:
            return False
    return path.suffix.lower() in SOURCE_EXTENSIONS


def _discover_files(project_path: str) -> list[Path]:
    root = Path(project_path)
    return [p for p in root.rglob("*") if p.is_file() and _should_index(p)]


async def index_project(project_path: str) -> int:
    """Index all source files under project_path. Returns total chunk count."""
    init_db()
    conn = get_db()
    pid = get_or_create_project(conn, project_path)
    conn.execute(
        "UPDATE projects SET indexed_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), pid),
    )
    conn.commit()
    conn.close()

    files = _discover_files(project_path)
    total = 0
    for file_path in files:
        total += await index_file(project_path, file_path)
    return total


async def index_file(project_path: str, file_path: Path) -> int:
    """Chunk, embed, and upsert a single file. Returns chunk count."""
    chunks = chunk_file(file_path)
    if not chunks:
        return 0

    collection = get_collection("code", project_path)

    try:
        rel = str(file_path.relative_to(project_path))
    except ValueError:
        rel = str(file_path)

    # Remove stale chunks for this file
    try:
        existing = collection.get(where={"file": rel})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    ids, embeddings, documents, metadatas = [], [], [], []
    for chunk in chunks:
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

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return len(chunks)


def clear_project_index(project_path: str) -> None:
    """Delete all indexed chunks for a project from ChromaDB."""
    collection = get_collection("code", project_path)
    all_ids = collection.get()["ids"]
    if all_ids:
        collection.delete(ids=all_ids)
