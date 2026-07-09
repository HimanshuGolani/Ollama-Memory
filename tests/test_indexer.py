from pathlib import Path
import pytest


def test_chunk_python_file(tmp_path):
    from indexer import chunk_file
    f = tmp_path / "example.py"
    f.write_text(
        "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
    )
    chunks = chunk_file(f)
    assert len(chunks) >= 1
    assert all("content" in c and "start_line" in c and "end_line" in c for c in chunks)
    full = "\n".join(c["content"] for c in chunks)
    assert "def foo" in full
    assert "def bar" in full


def test_chunk_unknown_extension_uses_sliding_window(tmp_path):
    from indexer import chunk_file
    f = tmp_path / "notes.txt"
    f.write_text("line\n" * 400)
    chunks = chunk_file(f)
    assert len(chunks) >= 2


def test_chunk_empty_file_returns_empty(tmp_path):
    from indexer import chunk_file
    f = tmp_path / "empty.py"
    f.write_text("")
    assert chunk_file(f) == []


def test_chunk_preserves_line_numbers(tmp_path):
    from indexer import chunk_file
    f = tmp_path / "sample.py"
    f.write_text("a = 1\nb = 2\ndef foo():\n    pass\n")
    chunks = chunk_file(f)
    assert chunks[0]["start_line"] == 1
    for c in chunks:
        assert c["start_line"] <= c["end_line"]
