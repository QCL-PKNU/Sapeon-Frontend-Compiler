from typing import Union

import numpy as np

from .tensor import Tensor


def load_tensor(tensor: Tensor, arg: Union[str, np.ndarray]) -> None:
    if isinstance(arg, str):
        tensor.load_txt(arg)
    elif isinstance(arg, np.ndarray):
        tensor.load_ndarray(arg)
    else:
        raise ValueError(
            f"tensor cannot load {type(arg).__name__} type arguments"
        )
