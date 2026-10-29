"""Use every CPU core: files in parallel processes, work inside a file in threads.

The heavy work runs in the compiled kernel, which releases Python's GIL, so
threads inside one file run truly in parallel. In a batch, each worker process
asks how many files are still unfinished and takes its share of the cores: while
many files remain, every file gets one core; at the end of the batch, the last
big files spread over the cores the others have freed.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

CPU = os.cpu_count() or 1
_PREFIX = "fotomedik-worker"

_pool: ThreadPoolExecutor | None = None
_fixed: int | None = None  # set_threads() override
_remaining = None  # multiprocessing.Value shared by batch workers: files not finished yet


def set_threads(n: int | None) -> None:
    """Force the number of threads per file (None: automatic)."""
    global _fixed
    _fixed = n


def attach_batch_counter(counter) -> None:
    global _remaining
    _remaining = counter


def threads() -> int:
    """Threads this file may use right now."""
    if _fixed is not None:
        return max(1, _fixed)
    if _remaining is not None:
        left = max(1, _remaining.value)
        return max(1, CPU // left)
    return CPU


def _get_pool() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=CPU, thread_name_prefix=_PREFIX)
    return _pool


def _after_fork() -> None:
    # a forked worker process inherits the pool object but not its threads
    global _pool
    _pool = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def pmap(fn, items: list, chunks: int | None = None) -> list:
    """fn over items, in order, spread over the available threads. Items are grouped
    into contiguous chunks (one per thread) so per-task overhead stays small."""
    n = len(items)
    t = min(threads(), n) if chunks is None else min(chunks, n)
    if threading.current_thread().name.startswith(_PREFIX):
        t = 1  # already inside a pool thread: waiting on the same pool could deadlock
    if t <= 1:
        return [fn(x) for x in items]
    size = -(-n // t)
    groups = [items[i : i + size] for i in range(0, n, size)]
    futs = [_get_pool().submit(lambda g: [fn(x) for x in g], g) for g in groups]
    out: list = []
    for f in futs:
        out.extend(f.result())
    return out
