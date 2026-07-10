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
    from db import get_db
    from watcher import start_watcher, stop_watcher
    conn = get_db()
    indexed = conn.execute(
        "SELECT path FROM projects WHERE indexed_at IS NOT NULL"
    ).fetchall()
    conn.close()
    observers = []
    for row in indexed:
        try:
            observers.append(start_watcher(row["path"]))
        except Exception:
            pass
    yield
    for obs in observers:
        stop_watcher(obs)


app = FastAPI(title="Ollama Brain", lifespan=lifespan)
app.include_router(context_router)
app.include_router(index_router)
app.include_router(remember_router)
app.include_router(status_router)
