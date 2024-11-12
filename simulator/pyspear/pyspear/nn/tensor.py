from typing import Tuple, Union, Optional
from enum import Enum
import numpy as np

from .dimension import Dimension


class DimFormat(Enum):
    NCHW = "NCHW"
    NHWC = "NHWC"
    NWHC = "NWHC"
    VECTOR = "VECTOR"


class DataType(Enum):
    float = "float32"
    double = "float64"
    float16 = "float16"
    uint8 = "uint8"
    sint8 = "int8"
    sint16 = "int16"


class Tensor:
    def __init__(
        self,
        dimension: Tuple[int, ...],
        dimfmt: Union[str, DimFormat] = None,
        dtype: Union[str, DataType] = None,
    ) -> None:
        self._dimension = Dimension(dimension)
        self._data: Optional[np.ndarray] = None

        if isinstance(dimfmt, str):
            dimfmt = DimFormat(dimfmt)
        self._dim_format: Optional[DimFormat] = dimfmt

        if isinstance(dtype, str):
            dtype = DataType(dtype)
        self._dtype: Optional[DataType] = dtype

    @property
    def dimension(self) -> Tuple[int, ...]:
        return self._dimension.dimension

    @property
    def data(self) -> Optional[np.ndarray]:
        return self._data

    @data.setter
    def data(self, np_array: np.ndarray) -> None:
        self._data = np_array

    @property
    def dim_format(self) -> DimFormat:
        return self._dim_format

    @property
    def datatype(self) -> DataType:
        return self._dtype

    @property
    def size(self) -> int:
        return self._dimension.size

    def load_txt(self, filepath: str) -> None:
        data = np.loadtxt(filepath)
        if data.size != self.size:
            raise ValueError(
                "The number of elements in the loaded file is not equal to the "
                "dimension"
            )
        self._data = data

    def load_ndarray(self, ndarray: np.ndarray) -> None:
        if ndarray.size != self.size:
            raise ValueError(
                "The number of elements in the loaded file is not equal to the "
                "dimension"
            )
        self._data = ndarray.flatten()
