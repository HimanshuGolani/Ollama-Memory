import asyncio
import time
from pathlib import Path
from threading import Thread

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class _IndexHandler(FileSystemEventHandler):
    def __init__(self, project_path: str, loop: asyncio.AbstractEventLoop):
        self._project = project_path
        self._loop = loop
        self._pending: dict[str, float] = {}
        self._debounce = 2.0

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def _schedule(self, path: str):
        self._pending[path] = time.monotonic() + self._debounce
        Thread(target=self._flush, args=(path,), daemon=True).start()

    def _flush(self, path: str):
        time.sleep(self._debounce + 0.1)
        due = self._pending.pop(path, 0)
        if time.monotonic() < due:
            return
        from indexer import index_file
        asyncio.run_coroutine_threadsafe(
            index_file(self._project, Path(path)),
            self._loop,
        )


def start_watcher(project_path: str) -> Observer:
    """Start a watchdog Observer on project_path. Returns the started Observer."""
    loop = asyncio.get_event_loop()
    handler = _IndexHandler(project_path, loop)
    observer = Observer()
    observer.schedule(handler, project_path, recursive=True)
    observer.start()
    return observer


def stop_watcher(observer: Observer) -> None:
    """Stop and join a previously started Observer."""
    observer.stop()
    observer.join()
