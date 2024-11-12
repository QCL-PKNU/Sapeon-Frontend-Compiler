from typing import Tuple, Union, List

from .node import Node
from .tuple_types import (
    padding_t,
    stride_t,
    window_t,
    dilation_t,
    kernel_size_t,
    scale_t,
    _int_scalar_or_tuple_2_t,
    _int_scalar_or_tuple_2_or_4_t,
    _float_scalar_or_tuple_2_t,
)


def _convert_to_4dim_int_tuple(
    _tuple: _int_scalar_or_tuple_2_or_4_t,
) -> Tuple[int, int, int, int]:
    if isinstance(_tuple, int):
        return (_tuple, _tuple, _tuple, _tuple)

    if isinstance(_tuple, tuple) and len(_tuple) == 2:
        height, width = _tuple
        return (height, height, width, width)

    # padding order : top, bottom, left, right
    if isinstance(_tuple, tuple) and len(_tuple) == 4:
        return _tuple

    raise ValueError("Not supported dimension")


def _convert_to_2dim_int_tuple(
    _tuple: _int_scalar_or_tuple_2_t,
) -> Tuple[int, int]:
    if isinstance(_tuple, int):
        return (_tuple, _tuple)

    if isinstance(_tuple, tuple) and len(_tuple) == 2:
        return _tuple

    raise ValueError("Not supported dimension")


def _convert_to_2dim_float_tuple(
    _tuple: _float_scalar_or_tuple_2_t,
) -> Tuple[float, float]:
    if isinstance(_tuple, float):
        return (_tuple, _tuple)

    if isinstance(_tuple, tuple) and len(_tuple) == 2:
        return _tuple

    raise ValueError("Not supported dimension")


def convert_kernel_size(kernel_sizes: kernel_size_t) -> Tuple[int, int]:
    return _convert_to_2dim_int_tuple(kernel_sizes)


def convert_padding(padding: padding_t) -> Tuple[int, int, int, int]:
    return _convert_to_4dim_int_tuple(padding)


def convert_stride(stride: stride_t) -> Tuple[int, int]:
    return _convert_to_2dim_int_tuple(stride)


def convert_window(window: window_t) -> Tuple[int, int]:
    return _convert_to_2dim_int_tuple(window)


def convert_dilation(dilation: dilation_t) -> Tuple[int, int]:
    return _convert_to_2dim_int_tuple(dilation)


def convert_scale(scale: scale_t) -> Tuple[float, float]:
    return _convert_to_2dim_float_tuple(scale)


def convert_nodes(nodes: Union[Node, List[Node]]) -> List[Node]:
    if isinstance(nodes, Node):
        return [nodes]
    return nodes


def assert_parent_nodes_inputs(child: Node, parents: List[Node]) -> None:
    if len(child.input_shapes) != len(parents):
        raise ValueError(
            f"The number of inputs in the {type(child).__name__} "
            "does not match the number of parent nodes"
        )

    for input_shape, parent in zip(child.input_shapes, parents):
        if input_shape.dimension != parent.output_shape.dimension:
            raise ValueError(
                f"{type(parent).__name__} ({parent.name}) output shape does not"
                f" support input shape of {type(child).__name__} ({child.name})"
            )
