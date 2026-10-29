import multiprocessing
import os

from fotomedik.cli import Workers


def _dies_in_worker(x):
    if multiprocessing.parent_process() is not None:
        os._exit(3)  # a worker process that cannot run (like a packaged app that fails to start)
    return x * 2


def test_broken_worker_processes_fall_back_to_this_process():
    with Workers(2, 4) as w:
        assert w.map(_dies_in_worker, [1, 2, 3, 4]) == [2, 4, 6, 8]
        assert w.ex is None
