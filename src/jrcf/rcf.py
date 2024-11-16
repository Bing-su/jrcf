from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict

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
from jpype.types import JArray, JFloat

if TYPE_CHECKING:
    import numpy.typing as npt

Array1D: TypeAlias = Sequence[float] | np.ndarray


class RCFArgs(TypedDict):
    forest: str | None
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
        parallel_execution_enabled: bool = False,
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
            If True, then the forest will create an internal threadpool.
            Forest updates and traversals will be submitted to this threadpool, and individual trees will be updated or traversed in parallel.
            For larger shingle sizes, dimensions, and number of trees, parallelization may improve throughput.
            We recommend users benchmark against their target use case. Defaults to False.
        thread_pool_size : int, optional
            The number of threads to use in the internal threadpool.
            if None, `Number of available processors - 1` is used. Defaults to None.
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
                builder = builder.threadPoolSize(self.thread_pool_size)

            if self.random_seed is not None:
                builder = builder.randomSeed(self.random_seed)

            self.forest = builder.build()

    def __rich_repr__(self):
        m = self.to_dict()
        for k, v in m.items():
            if k == "forest":
                continue
            yield k, v

    def __repr__(self) -> str:
        pair = [f"{k}={v!r}" for k, v in self.__rich_repr__()]
        return f"{self.__class__.__name__}({', '.join(pair)})"

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

    @staticmethod
    def _serialize_forest(forest: RandomCutForest) -> str:
        """
        Reference
        ---------
        https://github.com/aws/random-cut-forest-by-aws/blob/4.2.0-java/Java/examples/src/main/java/com/amazon/randomcutforest/examples/serialization/JsonExample.java
        """
        mapper = RandomCutForestMapper()
        mapper.setSaveExecutorContextEnabled(True)
        mapper.setSaveTreeStateEnabled(True)
        mapper.setPartialTreeStateEnabled(True)
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
        """
        Convert this instance to a dictionary.
        """
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
        """
        Create RandomCutForestModel from dictionary of arguments
        """
        if args.get("forest") is not None:
            args["forest"] = cls._deserialize_forest(args["forest"])  # type: ignore
        return cls(**args)

    def _convert_to_java_array(self, point: Array1D) -> JArray:
        return JArray.of(np.array(point), JFloat)

    def get_number_of_trees(self) -> int:
        """
        Returns
        -------
        int
            the number of trees in the forest.
        """
        return int(self.forest.getNumberOfTrees())

    def get_sample_size(self) -> int:
        """
        Returns
        -------
        int
            the sample size used by stream samplers in this forest.
        """
        return int(self.forest.getSampleSize())

    def get_shingle_size(self) -> int:
        """
        Returns
        -------
        int
            the shingle size used by the point store.
        """
        return int(self.forest.getShingleSize())

    def get_output_after(self) -> int:
        """
        Returns
        -------
        int
            the number of points required by stream samplers before results are returned.
        """
        return int(self.forest.getOutputAfter())

    def get_dimensions(self) -> int:
        """
        Returns
        -------
        int
            the number of dimensions in the data points accepted by this forest.
            i.e. input dimensions * shingle size
        """
        return int(self.forest.getDimensions())

    def get_time_decay(self) -> float:
        """
        Returns
        -------
        float
            the decay factor used by stream samplers in this forest.
        """
        return float(self.forest.getTimeDecay())

    def get_thread_pool_size(self) -> int:
        return int(self.forest.getThreadPoolSize())

    def transform_to_shingled_point(self, point: Array1D) -> npt.NDArray[np.float32]:
        """
        used for scoring and other function, expands to a shingled point in either case performs a clean copy

        Parameters
        ----------
        point: 1-d array-like

        Returns
        -------
        1-d np.array of np.float32
        """
        transformed = self.forest.transformToShingledPoint(
            self._convert_to_java_array(point)
        )
        "transformed is a java array of JFloat"
        return np.array(transformed)

    def is_output_ready(self) -> bool:
        return self.forest.isOutputReady()

    def score(self, point: Array1D) -> float:
        """
        Compute an anomaly score for the given point.

        Parameters
        ----------
        point: 1-d array-like
            A data point with input dimensions

        Returns
        -------
        float
            The anomaly score for the given point

        """
        arr = self._convert_to_java_array(point)
        score = self.forest.getAnomalyScore(arr)
        "score is JDouble"
        return float(score)

    def update(self, point: Array1D) -> None:
        """
        Update the model with the data point.

        Parameters
        ----------
        point: 1-d array-like
            A data point with input dimensions
        """
        self.forest.update(self._convert_to_java_array(point))

    def approximate_anomaly_score(self, point: Array1D) -> float:
        """
        Anomaly score evaluated sequentially with option of early stopping the early
        stopping parameter precision gives an approximate solution in the range
        (1-precision)*score(q) - precision, (1+precision)*score(q) + precision for the
        score of a point q. In this function z is hardcoded to 0.1. If this function
        is used, then not all the trees will be used in evaluation (but they have to
        be updated anyways, because they may be used for the next q). The advantage
        is that "almost certainly" anomalies/non-anomalies can be detected easily
        with few trees.

        Parameters
        ----------
        point: 1-d array-like
            A data point with input dimensions

        Returns
        -------
        float
            anomaly score with early stopping with z=0.1
        """
        arr = self._convert_to_java_array(point)
        score = self.forest.getApproximateAnomalyScore(arr)
        "score is JDouble"
        return float(score)
