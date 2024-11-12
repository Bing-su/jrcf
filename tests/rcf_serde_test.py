from __future__ import annotations

import pickle
from tempfile import TemporaryDirectory
from uuid import uuid4

import cloudpickle
import joblib
import numpy as np
import pytest
import skops.io
from hypothesis import given, settings
from hypothesis import strategies as st

from jrcf.rcf import RandomCutForestModel

NUM_DATA = 10


def assert_attrs(model1: RandomCutForestModel, model2: RandomCutForestModel):
    assert model1.dimensions == model2.dimensions
    assert model1.shingle_size == model2.shingle_size
    assert model1.num_trees == model2.num_trees
    assert model1.sample_size == model2.sample_size
    assert model1.output_after == model2.output_after
    assert model1.random_seed == model2.random_seed
    assert model1.parallel_execution_enabled == model2.parallel_execution_enabled
    assert model1.thread_pool_size == model2.thread_pool_size
    assert model1.lam == model2.lam
    assert model1.get_shingle_size() == model2.get_shingle_size()
    assert model1.get_thread_pool_size() == model2.get_thread_pool_size()


@pytest.mark.parametrize(
    "protocol", [*range(pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL + 1)]
)
@given(dim=st.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_pickling(dim: int, protocol: int):
    model = RandomCutForestModel(dimensions=dim, output_after=NUM_DATA)
    data = np.random.random((NUM_DATA, dim))
    for point in data:
        model.update(point)

    assert model.is_output_ready()

    anomaly = np.random.random(dim) + 1000
    score = model.score(anomaly)
    assert score >= 1.5

    pickled = pickle.dumps(model, protocol=protocol)
    unpickled = pickle.loads(pickled)  # noqa: S301  suspicious-pickle-usage

    assert_attrs(model, unpickled)

    assert unpickled.is_output_ready()

    score = unpickled.score(anomaly)
    assert score >= 1.5

    for point in data:
        unpickled.update(point)


@pytest.mark.parametrize(
    "protocol", [*range(pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL + 1)]
)
@given(dim=st.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_joblib(dim: int, protocol: int):
    model = RandomCutForestModel(dimensions=dim, output_after=NUM_DATA)
    data = np.random.random((NUM_DATA, dim))
    for point in data:
        model.update(point)

    assert model.is_output_ready()

    anomaly = np.random.random(dim) + 1000
    score = model.score(anomaly)
    assert score >= 1.5

    with TemporaryDirectory() as tmp:
        filename = f"{tmp}/{uuid4()}.joblib"
        joblib.dump(model, filename, protocol=protocol)
        unpickled = joblib.load(filename)

    assert_attrs(model, unpickled)

    assert unpickled.is_output_ready()

    score = unpickled.score(anomaly)
    assert score >= 1.5

    for point in data:
        unpickled.update(point)


@given(dim=st.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_cloudpickle(dim: int):
    model = RandomCutForestModel(dimensions=dim, output_after=NUM_DATA)
    data = np.random.random((NUM_DATA, dim))
    for point in data:
        model.update(point)

    assert model.is_output_ready()

    anomaly = np.random.random(dim) + 1000
    score = model.score(anomaly)
    assert score >= 1.5

    pickled = cloudpickle.dumps(model)
    unpickled = cloudpickle.loads(pickled)

    assert_attrs(model, unpickled)

    assert unpickled.is_output_ready()

    score = unpickled.score(anomaly)
    assert score >= 1.5

    for point in data:
        unpickled.update(point)


@pytest.mark.parametrize("compression", [0, 8, 12, 14])
@given(dim=st.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_skops(dim: int, compression: int):
    model = RandomCutForestModel(dimensions=dim, output_after=NUM_DATA)
    data = np.random.random((NUM_DATA, dim))
    for point in data:
        model.update(point)

    assert model.is_output_ready()

    anomaly = np.random.random(dim) + 1000
    score = model.score(anomaly)
    assert score >= 1.5

    pickled = skops.io.dumps(model, compression=compression)
    trusted = ["jrcf.rcf.RandomCutForestModel"]
    unpickled = skops.io.loads(pickled, trusted=trusted)

    assert_attrs(model, unpickled)

    assert unpickled.is_output_ready()

    score = unpickled.score(anomaly)
    assert score >= 1.5

    for point in data:
        unpickled.update(point)
