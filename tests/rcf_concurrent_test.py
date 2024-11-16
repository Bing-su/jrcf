from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

from jrcf.rcf import RandomCutForestModel


def create_and_update():
    model = RandomCutForestModel(dimensions=5, parallel_execution_enabled=True)

    scores = []
    for _ in range(5):
        data = np.random.random(5)
        scores.append(model.score(data))
        model.update(data)
    return scores


def test_thread_pool():
    tasks = []
    with ThreadPoolExecutor(5) as executor:
        for _ in range(5):
            task = executor.submit(create_and_update)
            tasks.append(task)

    result = [task.result() for task in tasks]
    assert all(isinstance(score, list) for score in result)
    assert isinstance(result[0][0], float)


def test_threadpool_with_shared_model():
    """
    이 테스트를 통과한다는 것이 스레드 안전을 의미하는 것은 아님
    """
    model = RandomCutForestModel(dimensions=5, parallel_execution_enabled=True)

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


def test_process_pool():
    tasks = []
    # https://jpype.readthedocs.io/en/stable/userguide.html#multiprocessing
    with ProcessPoolExecutor(3, mp_context=mp.get_context("spawn")) as executor:
        for _ in range(3):
            task = executor.submit(create_and_update)
            tasks.append(task)

    result = [task.result() for task in tasks]
    assert all(isinstance(score, list) for score in result)
    assert isinstance(result[0][0], float)


def test_process_pool_with_shared_model():
    """
    멀티프로세싱에서는 피클을 통해 객체를 전달하므로
    별도의 프로세스에서 동작하는 model.update는 원본 객체에 영향을 주지 않음
    이 테스트는 어쨌거나 멀티프로세싱에서 동작하는지 확인하기 위한 것
    """
    model = RandomCutForestModel(dimensions=5, parallel_execution_enabled=True)

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
