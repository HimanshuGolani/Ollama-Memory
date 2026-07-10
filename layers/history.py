import uuid
import httpx
from db import get_db, get_or_create_project, init_db
from embedder import embed
from chroma_client import get_collection
from config import settings

SUMMARIZE_EVERY = 10


async def record_query(query: str, project_path: str) -> None:
    """
    Records a raw query for the given project and triggers summarization
    every SUMMARIZE_EVERY queries.
    """
    init_db()
    conn = get_db()
    pid = get_or_create_project(conn, project_path)
    conn.execute(
        "INSERT INTO history_raw (project_id, query) VALUES (?, ?)", (pid, query)
    )
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
    """
    Summarizes a batch of queries using Ollama, with fallback to joining first 3.
    Stores the summary in both ChromaDB and the history_summaries SQLite table.
    """
    prompt = (
        "Summarize these coding assistant queries into 2-3 concise bullet points "
        "capturing the key topics and answers discussed. Be specific about code and files mentioned.\n\n"
        + "\n".join(f"- {q}" for q in queries)
    )
    summary = ""
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
    """
    Retrieves relevant past Q&A summaries for the given query.
    Returns [] if the history collection is empty (no embed call made).
    """
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
