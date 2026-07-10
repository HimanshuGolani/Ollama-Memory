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
