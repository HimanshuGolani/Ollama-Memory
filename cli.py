import asyncio
import json
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Ollama Brain — persistent memory for your codebase")
console = Console()

@app.command()
def serve():
    """Start the memory server."""
    import uvicorn
    from config import settings
    console.print(f"[green]Starting Ollama Brain on port {settings.server_port}[/green]")
    uvicorn.run("main:app", host="0.0.0.0", port=settings.server_port, reload=False)

@app.command()
def index(project: str = typer.Argument(..., help="Absolute path to project root")):
    """Index a project's source files into memory."""
    from db import init_db
    init_db()
    console.print(f"[cyan]Indexing {project}...[/cyan]")
    from indexer import index_project
    count = asyncio.run(index_project(project))
    console.print(f"[green]Done. Indexed {count} chunks.[/green]")

@app.command()
def remember(
    note: str = typer.Argument(..., help="Note text to save"),
    project: str = typer.Option(..., help="Absolute path to project root"),
):
    """Save a note about a project."""
    from db import init_db
    init_db()
    from layers.notes import save_note
    note_id = asyncio.run(save_note(note, project))
    console.print(f"[green]Note #{note_id} saved.[/green]")

@app.command()
def status():
    """Show indexed projects and stats."""
    from db import init_db, get_db
    init_db()
    conn = get_db()
    rows = conn.execute("SELECT path, name, indexed_at FROM projects").fetchall()
    conn.close()
    if not rows:
        console.print("[yellow]No projects indexed yet.[/yellow]")
        return
    table = Table("Project", "Name", "Last Indexed")
    for r in rows:
        table.add_row(r["path"], r["name"] or "-", r["indexed_at"] or "never")
    console.print(table)

@app.command()
def configure(project: str = typer.Argument(..., help="Absolute path to your project")):
    """Print the Continue.dev config snippet for this project."""
    from config import settings
    snippet = {
        "contextProviders": [
            {
                "name": "http",
                "params": {
                    "url": f"http://localhost:{settings.server_port}/context/code?project={project}",
                    "title": "Code Memory",
                    "description": "Relevant code chunks from the indexed codebase",
                    "displayTitle": "code",
                },
            },
            {
                "name": "http",
                "params": {
                    "url": f"http://localhost:{settings.server_port}/context/notes?project={project}",
                    "title": "Project Notes",
                    "description": "Curated architecture notes and decisions",
                    "displayTitle": "notes",
                },
            },
            {
                "name": "http",
                "params": {
                    "url": f"http://localhost:{settings.server_port}/context/history?project={project}",
                    "title": "Conversation History",
                    "description": "Relevant past Q&A about this project",
                    "displayTitle": "history",
                },
            },
        ],
        "slashCommands": [
            {
                "name": "remember",
                "description": "Save a note about this codebase",
                "step": "HttpSlashCommand",
                "params": {
                    "url": f"http://localhost:{settings.server_port}/remember"
                },
            }
        ],
    }
    console.print("\n[bold]Add this to ~/.continue/config.json:[/bold]\n")
    console.print(json.dumps(snippet, indent=2))

if __name__ == "__main__":
    app()
