from fastapi import APIRouter
from pydantic import BaseModel
from indexer import index_project, clear_project_index

router = APIRouter()


class ProjectBody(BaseModel):
    project: str


@router.post("/index")
async def trigger_index(body: ProjectBody):
    count = await index_project(body.project)
    return {"indexed_chunks": count, "project": body.project}


@router.delete("/index")
async def delete_index(body: ProjectBody):
    clear_project_index(body.project)
    return {"cleared": True, "project": body.project}
