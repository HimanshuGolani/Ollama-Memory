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
        file_name = meta.get("file", "unknown") if meta else "unknown"
        start = meta.get("start_line", 0) if meta else 0
        end = meta.get("end_line", 0) if meta else 0
        output.append({
            "name": f"{file_name}:{start}",
            "description": f"lines {start}–{end}",
            "content": doc,
        })
    return output
