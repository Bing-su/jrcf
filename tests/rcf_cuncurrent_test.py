from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

from jrcf.rcf import RandomCutForestModel


def test_threadpool():
    model = RandomCutForestModel(dimensions=5)

    tasks = []
    with ThreadPoolExecutor(5) as executor:
        for _ in range(5):
            data = np.random.random(5)
            task = executor.submit(model.score, data)
            tasks.append(task)
            executor.submit(model.update, data)

    scores = [task.result() for task in tasks]
    assert all(isinstance(score, float) for score in scores)
    assert all(score >= 0.0 for score in scores)


def test_processpool():
    model = RandomCutForestModel(dimensions=5)

    tasks = []
    # https://jpype.readthedocs.io/en/stable/userguide.html#multiprocessing
    with ProcessPoolExecutor(3, mp_context=mp.get_context("spawn")) as executor:
        for _ in range(3):
            data = np.random.random(5)
            task = executor.submit(model.score, data)
            tasks.append(task)
            executor.submit(model.update, data)

    scores = [task.result() for task in tasks]
    assert all(isinstance(score, float) for score in scores)
    assert all(score >= 0.0 for score in scores)
