from typing import Tuple
from functools import reduce


dimension_t = Tuple[int, ...]


class Dimension:
    def __init__(self, dimension: dimension_t):
        self._dimension: dimension_t = dimension
        self._size: int = None

    @property
    def dimension(self) -> dimension_t:
        return self._dimension

    @property
    def size(self) -> int:
        if self._size is None:
            self._size = reduce(lambda x, y: x * y, self._dimension)
        return self._size
