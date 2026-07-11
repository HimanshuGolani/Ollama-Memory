# Ollama Brain

Persistent memory for your local Ollama models. Gives AI assistants in VS Code and IntelliJ IDEA real knowledge of your codebase — across every session.

---

## The Problem

Ollama models forget everything when you close the chat. Ask about your auth flow today, come back tomorrow, and the model has no idea what you're talking about. It hallucinates file names, contradicts itself, and can't follow your architecture.

**Ollama Brain** fixes this by running a lightweight local server that indexes your code, stores your notes, and remembers your past questions — then surfaces all of it automatically through [Continue.dev](https://continue.dev) every time you ask a question.

---

## How It Works

Three memory layers, all local:

| Layer | What it stores | How it's used |
|---|---|---|
| **Code Index** | Every function, class, and file chunk in your project | Semantically searched on every query — relevant code is injected into the prompt |
| **Notes** | Architecture decisions, gotchas, important facts you want the model to remember | Persisted forever, retrieved by similarity |
| **History** | Auto-summarized records of past Q&A sessions | Gives the model context about what has already been discussed |

All data is stored on your machine (`~/.ollama-brain/`). Nothing leaves localhost.

---

## Prerequisites

- **Python 3.11+**
- **Ollama** running locally (`http://localhost:11434`)
- The following Ollama models pulled:
  ```
  ollama pull nomic-embed-text
  ollama pull llama3.2
  ```
- **Continue.dev** extension installed in VS Code or IntelliJ IDEA

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/HimanshuGolani/Ollama-Memory.git
cd "Ollama-Memory"
```

### 2. Create a virtual environment and install dependencies

**Windows (recommended — just double-click):**
```
start.bat
```
The batch file creates the venv, installs everything, and starts the server automatically.

**Manual (Windows / macOS / Linux):**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Starting the Server

```bash
python cli.py serve
```

Or on Windows, just double-click **`start.bat`**.

The server starts on port **11435** by default. You should see:

```
Starting Ollama Brain on port 11435
INFO:     Uvicorn running on http://0.0.0.0:11435
```

Keep this terminal open while you work.

---

## Step-by-Step Setup Guide

### Step 1 — Index your project

Point Ollama Brain at your codebase. It will scan every source file and build a searchable vector index:

```bash
python cli.py index "C:/path/to/your/project"
```

Example:
```bash
python cli.py index "C:/Users/Alice/projects/my-api"
```

Output:
```
Indexing C:/Users/Alice/projects/my-api...
Done. Indexed 847 chunks.
```

Large projects (1000+ files) take a minute or two on first run. After that, the file watcher automatically re-indexes any file you save.

### Step 2 — Configure Continue.dev

Run the configure command to get the exact snippet for your project:

```bash
python cli.py configure "C:/path/to/your/project"
```

This prints a JSON block like:

```json
{
  "contextProviders": [
    {
      "name": "http",
      "params": {
        "url": "http://localhost:11435/context/code?project=C:/path/to/your/project",
        "title": "Code Memory",
        "description": "Relevant code chunks from the indexed codebase",
        "displayTitle": "code"
      }
    },
    {
      "name": "http",
      "params": {
        "url": "http://localhost:11435/context/notes?project=C:/path/to/your/project",
        "title": "Project Notes",
        "description": "Curated architecture notes and decisions",
        "displayTitle": "notes"
      }
    },
    {
      "name": "http",
      "params": {
        "url": "http://localhost:11435/context/history?project=C:/path/to/your/project",
        "title": "Conversation History",
        "description": "Relevant past Q&A about this project",
        "displayTitle": "history"
      }
    }
  ],
  "slashCommands": [
    {
      "name": "remember",
      "description": "Save a note about this codebase",
      "step": "HttpSlashCommand",
      "params": {
        "url": "http://localhost:11435/remember"
      }
    }
  ]
}
```

### Step 3 — Paste into Continue.dev config

Open `~/.continue/config.json` and merge the `contextProviders` array and `slashCommands` array into the file.

**VS Code:** Press `Ctrl+Shift+P` → "Continue: Open config.json"

**IntelliJ IDEA:** Open the Continue sidebar → gear icon → "Open config.json"

After saving the config, the memory layers appear in Continue's `@` context menu.

### Step 4 — Use it

In Continue's chat input, type `@` to see your memory providers:

```
@code    — search indexed code chunks
@notes   — search your saved notes
@history — search past Q&A summaries
```

Example prompt:
```
@code @notes How does authentication work in this project?
```

Continue fetches the most relevant chunks and notes from Ollama Brain and injects them into the prompt automatically.

---

## Saving Notes

Tell the model things it should always remember about your project:

```bash
python cli.py remember "Auth uses JWT. Secrets are stored in .env. Never hardcode tokens." --project "C:/path/to/your/project"
```

Or use the `/remember` slash command inside Continue's chat:

```
/remember The database is Postgres 15, running on port 5432. The ORM is SQLAlchemy.
```

Notes are stored permanently and retrieved by semantic similarity on every query.

---

## CLI Reference

| Command | Description |
|---|---|
| `python cli.py serve` | Start the memory server (port 11435) |
| `python cli.py index <path>` | Index a project's source files |
| `python cli.py remember "<note>" --project <path>` | Save a note about a project |
| `python cli.py status` | List all indexed projects |
| `python cli.py configure <path>` | Print the Continue.dev config snippet for a project |

---

## Environment Variables

All settings have sensible defaults. Override with a `.env` file or shell environment:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_BRAIN_PORT` | `11435` | Port for the Ollama Brain server |
| `OLLAMA_BRAIN_DATA_DIR` | `~/.ollama-brain` | Where to store the database and vector index |
| `OLLAMA_BRAIN_EMBED_MODEL` | `nomic-embed-text` | Ollama model used for embeddings |
| `OLLAMA_BRAIN_CHAT_MODEL` | `llama3.2` | Ollama model used to summarize history |

Example `.env`:
```env
OLLAMA_BRAIN_PORT=12000
OLLAMA_BRAIN_EMBED_MODEL=nomic-embed-text
OLLAMA_BRAIN_CHAT_MODEL=llama3.1
```

---

## Supported File Types

Ollama Brain indexes source files from these extensions:

`.py` `.js` `.ts` `.jsx` `.tsx` `.java` `.kt` `.go` `.cs` `.cpp` `.c` `.h` `.rs` `.rb` `.php` `.swift` `.md` `.txt` `.yaml` `.yml` `.json` `.toml`

The following directories are automatically skipped:

`node_modules` `__pycache__` `.git` `.venv` `venv` `dist` `build` `.idea` `.vscode` `target` `bin` `obj`

---

## File Watcher (Auto Re-index)

Once a project is indexed, Ollama Brain watches for file saves and automatically re-indexes changed files within 2 seconds. No need to re-run `index` manually during a coding session.

Watching starts automatically for any previously-indexed projects when the server starts.

---

## Checking Status

```bash
python cli.py status
```

Shows a table of all indexed projects and when they were last indexed:

```
┌─────────────────────────────────┬──────────┬──────────────────────────┐
│ Project                         │ Name     │ Last Indexed             │
├─────────────────────────────────┼──────────┼──────────────────────────┤
│ C:/Users/Alice/projects/my-api  │ my-api   │ 2026-07-11T10:32:00+00:00│
└─────────────────────────────────┴──────────┴──────────────────────────┘
```

---

## Troubleshooting

**"unable to open database file" on first start**

Make sure `~/.ollama-brain` is writable. The server creates it automatically, but if the drive is full or permissions are wrong, it will fail.

**Embeddings are slow**

`nomic-embed-text` needs to be pulled: `ollama pull nomic-embed-text`. First run is slow while the model loads into memory; subsequent calls are fast.

**Continue.dev doesn't show `@code` / `@notes` / `@history`**

- Confirm the server is running (`python cli.py serve`)
- Check that the URLs in config.json use the correct project path (no trailing slash)
- Restart VS Code / IntelliJ after editing config.json

**History summarization isn't working**

History is summarized automatically every 10 queries using `llama3.2`. If that model isn't pulled (`ollama pull llama3.2`), the batch is silently skipped (queries are still recorded, just not summarized yet).

---

## Project Structure

```
Ollama-Memory/
├── cli.py              # CLI entry point (serve, index, remember, status, configure)
├── main.py             # FastAPI application
├── config.py           # Settings (env vars, defaults)
├── db.py               # SQLite layer
├── embedder.py         # Ollama embedding wrapper
├── chroma_client.py    # ChromaDB client wrapper
├── indexer.py          # File chunker + project indexer
├── watcher.py          # File system watcher (auto re-index on save)
├── layers/
│   ├── code.py         # Semantic code search
│   ├── notes.py        # Save + search notes
│   └── history.py      # Record queries + search summaries
├── routes/
│   ├── context.py      # POST /context/code|notes|history
│   ├── index_routes.py # POST|DELETE /index
│   ├── remember.py     # POST /remember
│   └── status.py       # GET /status
├── tests/              # 28 automated tests
├── requirements.txt
└── start.bat           # Windows one-click startup
```

---

## License

MIT
