from fastapi import APIRouter
from pydantic import BaseModel
from layers.notes import save_note

router = APIRouter()


class NoteBody(BaseModel):
    note: str
    project: str


@router.post("/remember")
async def remember(body: NoteBody):
    note_id = await save_note(body.note, body.project)
    return {"saved": True, "id": note_id}
