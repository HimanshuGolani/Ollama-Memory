import sqlite3

def test_init_db_creates_tables(tmp_data_dir):
    from db import init_db, get_db
    init_db()
    conn = get_db()
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert {"projects", "notes", "history_raw", "history_summaries"} <= tables
    conn.close()

def test_init_db_is_idempotent(tmp_data_dir):
    from db import init_db
    init_db()
    init_db()  # must not raise

def test_get_or_create_project(tmp_data_dir):
    from db import init_db, get_db, get_or_create_project
    init_db()
    conn = get_db()
    pid1 = get_or_create_project(conn, "/my/project")
    conn.commit()
    pid2 = get_or_create_project(conn, "/my/project")
    conn.commit()
    assert pid1 == pid2
    assert isinstance(pid1, int)
    conn.close()

def test_get_project_id_missing(tmp_data_dir):
    from db import init_db, get_db, get_project_id
    init_db()
    conn = get_db()
    assert get_project_id(conn, "/does/not/exist") is None
    conn.close()
