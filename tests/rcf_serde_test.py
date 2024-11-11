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


@pytest.mark.parametrize(
    "protocol", [*range(pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL + 1)]
)
@given(dim=st.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_pickling(dim: int, protocol: int):
    model = RandomCutForestModel(dimensions=dim)
    data = np.random.random((10, dim))
    for point in data:
        model.update(point)

    pickled = pickle.dumps(model, protocol=protocol)
    unpickled = pickle.loads(pickled)  # noqa: S301  suspicious-pickle-usage

    assert model.dimensions == unpickled.dimensions
    assert model.shingle_size == unpickled.shingle_size
    assert model.num_trees == unpickled.num_trees
    assert model.sample_size == unpickled.sample_size
    assert model.output_after == unpickled.output_after
    assert model.random_seed == unpickled.random_seed
    assert model.parallel_execution_enabled == unpickled.parallel_execution_enabled
    assert model.thread_pool_size == unpickled.thread_pool_size
    assert model.lam == unpickled.lam
    assert model.get_shingle_size() == unpickled.get_shingle_size()

    for point in data:
        unpickled.update(point)


@pytest.mark.parametrize(
    "protocol", [*range(pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL + 1)]
)
@given(dim=st.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_joblib(dim: int, protocol: int):
    model = RandomCutForestModel(dimensions=dim)
    data = np.random.random((10, dim))
    for point in data:
        model.update(point)

    with TemporaryDirectory() as tmp:
        filename = f"{tmp}/{uuid4()}.joblib"
        joblib.dump(model, filename, protocol=protocol)
        unpickled = joblib.load(filename)

    assert model.dimensions == unpickled.dimensions
    assert model.shingle_size == unpickled.shingle_size
    assert model.num_trees == unpickled.num_trees
    assert model.sample_size == unpickled.sample_size
    assert model.output_after == unpickled.output_after
    assert model.random_seed == unpickled.random_seed
    assert model.parallel_execution_enabled == unpickled.parallel_execution_enabled
    assert model.thread_pool_size == unpickled.thread_pool_size
    assert model.lam == unpickled.lam
    assert model.get_shingle_size() == unpickled.get_shingle_size()

    for point in data:
        unpickled.update(point)


@given(dim=st.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_cloudpickle(dim: int):
    model = RandomCutForestModel(dimensions=dim)
    data = np.random.random((10, dim))
    for point in data:
        model.update(point)

    pickled = cloudpickle.dumps(model)
    unpickled = cloudpickle.loads(pickled)

    assert model.dimensions == unpickled.dimensions
    assert model.shingle_size == unpickled.shingle_size
    assert model.num_trees == unpickled.num_trees
    assert model.sample_size == unpickled.sample_size
    assert model.output_after == unpickled.output_after
    assert model.random_seed == unpickled.random_seed
    assert model.parallel_execution_enabled == unpickled.parallel_execution_enabled
    assert model.thread_pool_size == unpickled.thread_pool_size
    assert model.lam == unpickled.lam
    assert model.get_shingle_size() == unpickled.get_shingle_size()

    for point in data:
        unpickled.update(point)


@pytest.mark.parametrize("compression", [0, 8, 12, 14])
@given(dim=st.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_skops(dim: int, compression: int):
    model = RandomCutForestModel(dimensions=dim)
    data = np.random.random((10, dim))
    for point in data:
        model.update(point)

    pickled = skops.io.dumps(model, compression=compression)
    trusted = ["jrcf.rcf.RandomCutForestModel"]
    unpickled = skops.io.loads(pickled, trusted=trusted)

    assert model.dimensions == unpickled.dimensions
    assert model.shingle_size == unpickled.shingle_size
    assert model.num_trees == unpickled.num_trees
    assert model.sample_size == unpickled.sample_size
    assert model.output_after == unpickled.output_after
    assert model.random_seed == unpickled.random_seed
    assert model.parallel_execution_enabled == unpickled.parallel_execution_enabled
    assert model.thread_pool_size == unpickled.thread_pool_size
    assert model.lam == unpickled.lam
    assert model.get_shingle_size() == unpickled.get_shingle_size()

    for point in data:
        unpickled.update(point)
