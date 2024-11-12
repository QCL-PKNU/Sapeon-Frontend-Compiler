from typing import List, Tuple, Optional, Union

from .node import Node
from .activation import Activation
from .tuple_types import (
    padding_t,
    window_t,
    stride_t,
)
from .node_util import (
    convert_padding,
    convert_window,
    convert_stride,
    convert_nodes,
    assert_parent_nodes_inputs,
)


class Lavgpool(Node):
    def __init__(
        self,
        node_id: int,
        name: str,
        input_shapes: List[Tuple[int, ...]],
        output_shape: Tuple[int, ...],
        window: window_t,
        activation: Optional[Activation] = None,
        stride: Optional[stride_t] = None,
        padding: Optional[padding_t] = 0,
    ):
        if len(input_shapes) > 1:
            raise ValueError(
                f"{type(self).__name__} node cannot accept more than one input"
            )
        super().__init__(
            node_id=node_id,
            name=name,
            input_shapes=input_shapes,
            output_shape=output_shape,
            activation=activation,
        )

        self.window: Tuple[int, int] = convert_window(window)
        if stride is None:
            stride = (self.window[0], self.window[1])
        self.stride: Tuple[int, int] = convert_stride(stride)
        self.padding: Tuple[int, int, int, int] = convert_padding(padding)

    def set_parents(self, nodes: Union[Node, List[Node]]) -> None:
        nodes = convert_nodes(nodes)
        assert_parent_nodes_inputs(self, nodes)

        self.parents.clear()
        self.parents.extend(nodes)
