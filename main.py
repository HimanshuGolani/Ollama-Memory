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
