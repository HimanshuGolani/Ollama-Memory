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
