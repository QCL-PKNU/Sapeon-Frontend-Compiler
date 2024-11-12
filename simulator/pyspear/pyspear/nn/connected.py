from typing import List, Tuple, Union, Optional

import numpy as np

from .node import Node
from .activation import Activation
from .tensor import Tensor
from .node_util import convert_nodes, assert_parent_nodes_inputs
from .matmul_util import load_tensor


class Connected(Node):
    def __init__(
        self,
        node_id: int,
        name: str,
        input_shapes: List[Tuple[int, ...]],
        output_shape: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        activation: Optional[Activation] = None,
        weight_dtype: Optional[str] = "float32",
        bias_dtype: Optional[str] = "float32",
    ) -> None:
        if len(input_shapes) > 1:
            raise ValueError(
                f"{type(self).__name__} cannot accept more than one input"
            )
        if input_shapes[0][1] != in_channels:
            raise ValueError(
                f"{type(self).__name__}'s input dimension is incorrect with "
                f"dims : {input_shapes[0]}"
            )
        if output_shape[1] != out_channels:
            raise ValueError(
                f"{type(self).__name__}'s output dimension is incorrect with "
                f"dims : {output_shape}"
            )

        super().__init__(
            node_id=node_id,
            name=name,
            input_shapes=input_shapes,
            output_shape=output_shape,
            activation=activation,
        )
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.weights: Tensor = Tensor(
            dimension=(out_channels, in_channels, 1, 1),
            dtype=weight_dtype,
        )
        self.bias: Tensor = Tensor(
            dimension=(out_channels,),
            dtype=bias_dtype,
            dimfmt="VECTOR",
        )

    def set_parents(self, nodes: Union[Node, List[Node]]) -> None:
        nodes = convert_nodes(nodes)
        assert_parent_nodes_inputs(self, nodes)

        self.parents.clear()
        self.parents.extend(nodes)

    def load_filter_weight(self, arg: Union[str, np.ndarray]) -> None:
        load_tensor(self.weights, arg)

    def load_filter_bias(self, arg: Union[str, np.ndarray]) -> None:
        load_tensor(self.bias, arg)
