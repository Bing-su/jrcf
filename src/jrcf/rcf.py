from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from typing import Any, TypedDict

import jpype
import numpy as np

# java imports
from com.amazon.randomcutforest import (  # type: ignore [reportMissingImports]
    RandomCutForest,
)
from com.amazon.randomcutforest.state import (  # type: ignore [reportMissingImports]
    RandomCutForestMapper,
    RandomCutForestState,
)
from com.fasterxml.jackson.databind import (  # type: ignore [reportMissingImports]
    ObjectMapper,
)
from jpype.types import JArray, JDouble


class RCFArgs(TypedDict):
    forest: RandomCutForest | None
    dimensions: int
    shingle_size: int
    num_trees: int
    sample_size: int
    output_after: int | None
    random_seed: int | None
    parallel_execution_enabled: bool
    thread_pool_size: int | None
    lam: float | None


class RandomCutForestModel:
    """
    Random Cut Forest Python Binding around the AWS Random Cut Forest Official Java version:
    https://github.com/aws/random-cut-forest-by-aws
    """

    def __init__(  # noqa: PLR0913
        self,
        forest: RandomCutForest | None = None,
        *,
        dimensions: int = 1,
        shingle_size: int = 8,
        num_trees: int = 50,
        sample_size: int = 256,
        output_after: int | None = None,
        random_seed: int | None = None,
        parallel_execution_enabled: bool = True,
        thread_pool_size: int | None = None,
        lam: float | None = None,
    ):
        """
        Initialize the RandomCutForest model.

        Parameters
        ----------
        forest : RandomCutForest, optional
            A pre-trained RandomCutForest model. Used for pickling. Defaults to None.
        dimensions : int, optional
            The number of dimensions in the input data. Defaults to 1.
        shingle_size : int, optional
            The number of contiguous observations across all the input variables that would be used for analysis. Defaults to 8.
        num_trees : int, optional
            The number of trees in this forest. Defaults to 50.
        sample_size : int, optional
            The sample size used by stream samplers in this forest. Defaults to 256.
        output_after : int, optional
            The number of points required by stream samplers before results are returned. If None, `0.25 * sample_size` is used. Defaults to None.
        random_seed : int, optional
            A seed value used to initialize the random number generators in this forest. Defaults to None.
        parallel_execution_enabled : bool, optional
            If True, then the forest will create an internal threadpool. Forest updates and traversals will be submitted to this threadpool, and individual trees will be updated or traversed in parallel. For larger shingle sizes, dimensions, and number of trees, parallelization may improve throughput. We recommend users benchmark against their target use case. Defaults to True.
        thread_pool_size : int, optional
            The number of threads to use in the internal threadpool. Defaults to None.
        lam : float, optional
            The decay factor used by stream samplers in this forest.
            see: https://github.com/aws/random-cut-forest-by-aws/tree/4.2.0-java/Java#choosing-a-timedecay-value-for-your-application
            If None, default value is `1.0 / (10 * sample_size)`. Defaults to None.

        References
        ----------
        https://github.com/aws/random-cut-forest-by-aws/tree/4.2.0-java/Java
        """
        self.dimensions = dimensions
        self.shingle_size = shingle_size
        self.num_trees = num_trees
        self.sample_size = sample_size
        self.output_after = (
            output_after if output_after is not None else sample_size // 4
        )
        self.random_seed = random_seed
        self.parallel_execution_enabled = parallel_execution_enabled
        self.thread_pool_size = thread_pool_size
        self.lam = lam if lam is not None else 1.0 / (10 * sample_size)

        if forest is not None:
            self.forest = forest
        else:
            builder = (
                RandomCutForest.builder()
                .numberOfTrees(self.num_trees)
                .sampleSize(self.sample_size)
                .dimensions(self.dimensions * self.shingle_size)
                .shingleSize(self.shingle_size)
                .storeSequenceIndexesEnabled(True)
                .centerOfMassEnabled(True)
                .parallelExecutionEnabled(self.parallel_execution_enabled)
                .timeDecay(self.lam)
                .outputAfter(self.output_after)
                .internalShinglingEnabled(True)
            )
            if self.thread_pool_size is not None:
                builder.threadPoolSize(self.thread_pool_size)

            if self.random_seed is not None:
                builder = builder.randomSeed(self.random_seed)

            self.forest = builder.build()

    @staticmethod
    def _serialize_forest(forest: RandomCutForest) -> str:
        mapper = RandomCutForestMapper()
        mapper.setSaveExecutorContextEnabled(True)
        json_mapper = ObjectMapper()
        forest_state = mapper.toState(forest)
        json_string = json_mapper.writeValueAsString(forest_state)
        return str(json_string)

    @staticmethod
    def _deserialize_forest(string: str) -> RandomCutForest:
        mapper = RandomCutForestMapper()
        json_mapper = ObjectMapper()
        forest_state = json_mapper.readValue(string, RandomCutForestState)
        return mapper.toModel(forest_state)

    def to_dict(self) -> RCFArgs:
        return {
            "forest": self._serialize_forest(self.forest),
            "dimensions": self.dimensions,
            "shingle_size": self.shingle_size,
            "num_trees": self.num_trees,
            "sample_size": self.sample_size,
            "output_after": self.output_after,
            "random_seed": self.random_seed,
            "parallel_execution_enabled": self.parallel_execution_enabled,
            "thread_pool_size": self.thread_pool_size,
            "lam": self.lam,
        }

    @classmethod
    def from_dict(cls, args: RCFArgs) -> RandomCutForestModel:
        if args.get("forest") is not None:
            args["forest"] = cls._deserialize_forest(args["forest"])  # type: ignore
        return cls(**args)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        forest = state.pop("forest")
        state = copy.deepcopy(state)
        state["forest"] = self._serialize_forest(forest)
        return state

    def __setstate__(self, state: dict[str, Any]):
        json_string: str = state["forest"]
        state["forest"] = self._deserialize_forest(json_string)
        self.__dict__.update(state)

    def _convert_to_java_array(self, point: Sequence[float]) -> JArray:
        return JArray.of(np.array(point), JDouble)

    def score(self, point: Sequence[float]) -> float:
        """
        Compute an anomaly score for the given point.

        Parameters
        ----------
        point: Sequence[float]
            A data point with shingle size

        Returns
        -------
        float
            The anomaly score for the given point

        """
        return self.forest.getAnomalyScore(self._convert_to_java_array(point))

    def update(self, point: Sequence[float]):
        """
        Update the model with the data point.

        Parameters
        ----------
        point: Sequence[float]
            Point with shingle size
        """
        self.forest.update(self._convert_to_java_array(point))

    def impute(self, point: Sequence[float]) -> list[float]:
        """
        Given a point with missing values, return a new point with the missing values imputed. Each tree in the forest
        individual produces an imputed value. For 1-dimensional points, the median imputed value is returned. For
        points with more than 1 dimension, the imputed point with the 25th percentile anomaly score is returned.

        Parameters
        ----------
        point: List[float]
            The point with shingle size

        Returns
        -------
        List[float]
            The imputed point.
        """

        num_missing = np.isnan(point).sum()
        if num_missing == 0:
            return list(point)
        missing_index = np.argwhere(np.isnan(point)).flatten()
        return list(self.forest.imputeMissingValues(point, missing_index))

    def get_shingle_size(self) -> int:
        """
        Returns
        -------
        int
            Shingle size of random cut trees.
        """
        return self.forest.getDimensions()

    def get_attribution(
        self, point: Sequence[float]
    ) -> tuple[list[float], list[float]]:
        point = self._convert_to_java_array(point)
        try:
            attribution_di_vec: Any = self.forest.getAnomalyAttribution(point)
            low: list[float] = list(attribution_di_vec.low)
            high: list[float] = list(attribution_di_vec.high)
        except jpype.JException as exception:
            logging.info("Error when loading the model: %s", exception.message())
            logging.info("Stack track: %s", exception.stacktrace())
            # Throw it back
            raise
        else:
            return low, high
