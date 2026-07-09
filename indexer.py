import re
from pathlib import Path

LANGUAGE_PATTERNS: dict[str, str] = {
    ".py":   r"^(def |class |\s{0,4}def |\s{0,4}class )",
    ".js":   r"^(function |const |class |export )",
    ".ts":   r"^(function |const |class |export |interface |type )",
    ".java": r"^\s*(public|private|protected|static|class|interface|enum)\s",
    ".kt":   r"^\s*(fun |class |object |interface )",
    ".go":   r"^func ",
    ".cs":   r"^\s*(public|private|protected|static|class|interface|namespace)\s",
}

SKIP_DIRS: set[str] = {
    "node_modules", "__pycache__", ".git", ".venv", "venv",
    "dist", "build", ".idea", ".vscode", "target", "bin", "obj",
}

SOURCE_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".kt", ".go",
    ".cs", ".cpp", ".c", ".h", ".rs", ".rb", ".php", ".swift",
    ".md", ".txt", ".yaml", ".yml", ".json", ".toml",
}

WINDOW_CHARS = 1200
OVERLAP_LINES = 3


def chunk_file(path: Path) -> list[dict]:
    """
    Split a file into chunks, using language-specific patterns if available,
    otherwise using a sliding window approach.

    Returns:
        List of dicts with keys: "content", "start_line", "end_line"
        Empty list if file is empty or unreadable.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    if not text.strip():
        return []

    pattern = LANGUAGE_PATTERNS.get(path.suffix.lower())
    raw = _split_by_pattern(text, pattern) if pattern else _sliding_window(text)

    # Filter out chunks with empty content after stripping
    return [c for c in raw if c["content"].strip()]


def _split_by_pattern(text: str, pattern: str) -> list[dict]:
    """
    Split text by language-specific pattern (functions/classes).
    When a line matches the pattern and we have accumulated lines,
    start a new chunk.
    """
    lines = text.split("\n")
    chunks: list[dict] = []
    current: list[str] = []
    start = 0

    for i, line in enumerate(lines):
        if re.match(pattern, line) and current:
            # Pattern matched and we have content: save current chunk, start new one
            chunks.append(_make_chunk(current, start))
            current = [line]
            start = i
        else:
            # Just accumulate the line
            current.append(line)

    # Don't forget the last chunk
    if current:
        chunks.append(_make_chunk(current, start))

    return _merge_tiny(chunks)


def _sliding_window(text: str) -> list[dict]:
    """
    Split text using a sliding window approach: accumulate lines until
    we reach WINDOW_CHARS characters, then create a chunk with OVERLAP_LINES overlap.
    """
    lines = text.split("\n")
    chunks: list[dict] = []
    current: list[str] = []
    chars = 0
    start = 0

    for i, line in enumerate(lines):
        current.append(line)
        chars += len(line) + 1  # +1 for newline

        if chars >= WINDOW_CHARS:
            # Window full: save chunk
            chunks.append(_make_chunk(current, start))

            # Prepare overlap
            overlap = current[-OVERLAP_LINES:]
            current = list(overlap)
            chars = sum(len(l) + 1 for l in current)
            start = i - len(overlap) + 1

    # Don't forget the final chunk
    if current:
        chunks.append(_make_chunk(current, start))

    return chunks


def _make_chunk(lines: list[str], start_idx: int) -> dict:
    """Create a chunk dict from lines and starting index."""
    return {
        "content": "\n".join(lines),
        "start_line": start_idx + 1,  # 1-based
        "end_line": start_idx + len(lines),
    }


def _merge_tiny(chunks: list[dict], min_chars: int = 50) -> list[dict]:
    """
    Merge chunks shorter than min_chars into the previous chunk.
    """
    merged: list[dict] = []
    for chunk in chunks:
        if merged and len(merged[-1]["content"]) < min_chars:
            # Previous chunk is too small: merge this one into it
            merged[-1]["content"] += "\n" + chunk["content"]
            merged[-1]["end_line"] = chunk["end_line"]
        else:
            # Either no previous chunk or it's large enough: add as new chunk
            merged.append(chunk)
    return merged
