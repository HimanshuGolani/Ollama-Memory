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
