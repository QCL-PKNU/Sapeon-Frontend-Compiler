from typing import List, Tuple, Optional, Union

from .node import Node
from .activation import Activation
from .tuple_types import (
    padding_t,
)
from .node_util import (
    convert_padding,
    convert_window,
    convert_nodes,
    assert_parent_nodes_inputs,
)


class Gavgpool(Node):
    def __init__(
        self,
        node_id: int,
        name: str,
        input_shapes: List[Tuple[int, ...]],
        output_shape: Tuple[int, ...],
        activation: Optional[Activation] = None,
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

        self.padding: Tuple[int, int, int, int] = convert_padding(padding)
        self.window: Tuple[int, int] = None
        if len(input_shapes[0]) == 4:
            self.window = convert_window(
                (input_shapes[0][2], input_shapes[0][3])
            )
        else:
            raise ValueError(
                f"{type(self).__name__} should provided 4 dimension input shape"
            )

    def set_parents(self, nodes: Union[Node, List[Node]]) -> None:
        nodes = convert_nodes(nodes)
        assert_parent_nodes_inputs(self, nodes)

        self.parents.clear()
        self.parents.extend(nodes)
