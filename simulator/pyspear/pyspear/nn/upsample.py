from typing import List, Tuple, Optional, Union

from .node import Node
from .activation import Activation
from .tuple_types import stride_t
from .node_util import (
    convert_stride,
    convert_nodes,
    assert_parent_nodes_inputs,
)


class Upsample(Node):
    def __init__(
        self,
        node_id: int,
        name: str,
        input_shapes: List[Tuple[int, ...]],
        output_shape: Tuple[int, ...],
        stride: stride_t,
        activation: Optional[Activation] = None,
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

        self.stride: Tuple[int, int] = convert_stride(stride)

    def set_parents(self, nodes: Union[Node, List[Node]]) -> None:
        nodes = convert_nodes(nodes)
        assert_parent_nodes_inputs(self, nodes)

        self.parents.clear()
        self.parents.extend(nodes)
