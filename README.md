# JRCF

Java Random Cut Forest

https://github.com/aws/random-cut-forest-by-aws 저장소의 `python_rcf_wrapper`를 참고하여 파이썬에서 실행할 수 있도록 구성한 Random Cut Forest 알고리즘입니다.

## Requirements

- Java 8 or later

## Installation

```sh
pip install jrcf
```

## Usage

```python
import numpy as np
from tqdm.auto import tqdm

from jrcf.rcf import RandomCutForestModel

dim = 5
forest = RandomCutForestModel(dimensions=dim)
TEST_DATA = np.random.normal(size=(100000, dim))

for point in tqdm(TEST_DATA):
    score = forest.score(point)
    forest.update(point)

pp = [999] * dim
print(forest.score(pp))
```
