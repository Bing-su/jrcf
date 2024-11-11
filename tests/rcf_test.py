from __future__ import annotations

import json
from collections import UserList

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jrcf.rcf import RandomCutForestModel


@given(
    dimensions=st.integers(min_value=1, max_value=100),
    shingle_size=st.integers(min_value=1, max_value=20),
    num_trees=st.integers(min_value=1, max_value=100),
    sample_size=st.integers(min_value=4, max_value=512),
    output_after=st.one_of(st.none(), st.integers(min_value=1, max_value=512)),
    random_seed=st.one_of(
        st.none(), st.integers(min_value=-(2**32) + 1, max_value=2**32 - 1)
    ),
    parallel_execution_enabled=st.booleans(),
    thread_pool_size=st.one_of(st.none(), st.integers(min_value=1, max_value=8)),
    lam=st.one_of(st.none(), st.floats(min_value=0, max_value=1)),
)
@settings(deadline=None)
def test_rcf_init(  # noqa: PLR0913
    dimensions: int,
    shingle_size: int,
    num_trees: int,
    sample_size: int,
    output_after: int | None,
    random_seed: int | None,
    parallel_execution_enabled: bool,
    thread_pool_size: int | None,
    lam: float | None,
):
    try:
        model = RandomCutForestModel(
            dimensions=dimensions,
            shingle_size=shingle_size,
            num_trees=num_trees,
            sample_size=sample_size,
            output_after=output_after,
            random_seed=random_seed,
            parallel_execution_enabled=parallel_execution_enabled,
            thread_pool_size=thread_pool_size,
            lam=lam,
        )
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

    assert model.forest is not None
    assert model.get_shingle_size() == dimensions * shingle_size

    dump = model.to_dict()
    assert dump["dimensions"] == model.dimensions
    assert dump["shingle_size"] == model.shingle_size
    assert dump["num_trees"] == model.num_trees
    assert dump["sample_size"] == model.sample_size
    assert dump["output_after"] == model.output_after
    assert dump["random_seed"] == model.random_seed
    assert dump["parallel_execution_enabled"] == model.parallel_execution_enabled
    assert dump["thread_pool_size"] == model.thread_pool_size
    assert dump["lam"] == model.lam

    try:
        string = json.dumps(dump)
    except Exception as e:
        pytest.fail(f"json.dumps failed: {e}")

    loaded_json = json.loads(string)

    loaded = RandomCutForestModel.from_dict(loaded_json)
    assert loaded.forest is not None
    assert loaded.get_shingle_size() == dimensions * shingle_size

    assert model.dimensions == loaded.dimensions
    assert model.shingle_size == loaded.shingle_size
    assert model.num_trees == loaded.num_trees
    assert model.sample_size == loaded.sample_size
    assert model.output_after == loaded.output_after
    assert model.random_seed == loaded.random_seed
    assert model.parallel_execution_enabled == loaded.parallel_execution_enabled
    assert model.thread_pool_size == loaded.thread_pool_size
    assert model.lam == loaded.lam


@given(dim=st.integers(min_value=1, max_value=100))
@settings(deadline=None)
def test_train(dim: int):
    model = RandomCutForestModel(dimensions=dim)
    data = np.random.random((10, dim))
    for point in data:
        score = model.score(point)
        model.update(point)
        assert isinstance(score, float)
        assert score >= 0.0


def test_input_type():
    model = RandomCutForestModel(dimensions=5)

    arr = np.random.random(5)
    model.score(arr)
    model.update(arr)

    arr2 = np.random.random(5).tolist()
    model.score(arr2)
    model.update(arr2)

    arr3 = tuple(np.random.random(5).tolist())
    model.score(arr3)
    model.update(arr3)

    arr4 = UserList(np.random.random(5).tolist())
    model.score(arr4)
    model.update(arr4)


def test_repr():
    model = RandomCutForestModel(dimensions=5)
    repr_str = repr(model)

    assert "RandomCutForestModel(" in repr_str
    assert "dimensions=5" in repr_str
