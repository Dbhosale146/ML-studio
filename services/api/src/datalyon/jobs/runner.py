"""Background job runner. Work never runs in the request thread.

The interface lets a process pool or a queue-backed worker replace the local thread pool later
(see docs/adr/0006-agent-runs-in-threads.md for why threads are used today)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Protocol

log = logging.getLogger("datalyon.jobs")


class JobRunner(Protocol):
    def submit(self, name: str, fn: Callable[[], Any]) -> Future[Any]: ...

    def shutdown(self) -> None: ...


class ThreadJobRunner:
    def __init__(self, max_workers: int = 3) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="datalyon-job")

    def submit(self, name: str, fn: Callable[[], Any]) -> Future[Any]:
        def wrapped() -> Any:
            try:
                return fn()
            except Exception:
                log.exception("job %s crashed", name)
                raise

        return self._pool.submit(wrapped)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
