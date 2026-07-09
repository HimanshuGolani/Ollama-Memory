import hashlib
from typing import Optional
import chromadb
from config import settings

_client: Optional[chromadb.PersistentClient] = None


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
