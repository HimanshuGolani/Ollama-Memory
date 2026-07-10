import uuid
from db import get_db, get_or_create_project, init_db
from embedder import embed
from chroma_client import get_collection


async def save_note(note: str, project_path: str) -> int:
    """
    Save a note to both SQLite and ChromaDB.

    1. init_db() to ensure schema exists
    2. Embed the note text
    3. Generate a UUID for ChromaDB ID
    4. Upsert to ChromaDB "notes" collection
    5. Insert into SQLite notes table
    6. Return the SQLite note ID
    """
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
    """
    Semantically search notes for a project.

    1. Get the "notes" collection for this project
    2. If collection is empty, return [] (don't call embed)
    3. Embed the query
    4. Query ChromaDB with n_results = min(n_results, collection.count())
    5. For each result, look up the SQLite record by chroma_id
    6. Return list of dicts with name, description, and content
    """
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
