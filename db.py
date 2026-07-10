import sqlite3
from pathlib import Path
from config import settings


def get_db() -> sqlite3.Connection:
    """
    Opens (creating if needed) the SQLite database at settings.data_dir / "memory.db"
    Sets row_factory = sqlite3.Row and PRAGMA foreign_keys = ON
    Returns open connection (caller is responsible for closing)
    """
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db_path = settings.data_dir / "memory.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """
    Creates all four tables if they don't exist (idempotent)
    Opens and closes its own connection
    """
    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            name TEXT,
            indexed_at DATETIME,
            config_json TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            tags TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            chroma_id TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS history_raw (
            id INTEGER PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            query TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS history_summaries (
            id INTEGER PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            summary TEXT NOT NULL,
            chroma_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def get_or_create_project(conn: sqlite3.Connection, path: str) -> int:
    """
    Returns projects.id for the given path, inserting if not present
    Uses Path(path).name as the project name on insert
    Does NOT commit — caller commits
    """
    # Try to find existing project
    cursor = conn.execute("SELECT id FROM projects WHERE path = ?", (path,))
    row = cursor.fetchone()

    if row:
        return row["id"]

    # Insert new project
    project_name = Path(path).name
    cursor = conn.execute(
        "INSERT INTO projects (path, name) VALUES (?, ?)",
        (path, project_name)
    )

    return cursor.lastrowid


def get_project_id(conn: sqlite3.Connection, path: str) -> int | None:
    """
    Returns projects.id for the given path, or None if not found
    """
    cursor = conn.execute("SELECT id FROM projects WHERE path = ?", (path,))
    row = cursor.fetchone()
    return row["id"] if row else None
