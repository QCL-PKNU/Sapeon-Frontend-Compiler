from typing import List, Tuple, Optional, Union

from .node import Node
from .activation import Activation
from .node_util import convert_nodes, assert_parent_nodes_inputs


class Route(Node):
    def __init__(
        self,
        node_id: int,
        name: str,
        input_shapes: List[Tuple[int, ...]],
        output_shape: Tuple[int, ...],
        activation: Optional[Activation] = None,
    ) -> None:
        if len(input_shapes) < 2:
            raise ValueError(
                "The input layers of the Route are not set to more than one"
            )
        super().__init__(
            node_id=node_id,
            name=name,
            input_shapes=input_shapes,
            output_shape=output_shape,
            activation=activation,
        )

    def set_parents(self, nodes: Union[Node, List[Node]]) -> None:
        nodes = convert_nodes(nodes)
        assert_parent_nodes_inputs(self, nodes)

        self.parents.clear()
        self.parents.extend(nodes)
