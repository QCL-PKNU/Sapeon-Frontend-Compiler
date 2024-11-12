from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union

from .dimension import Dimension
from .activation import Activation


class Node(ABC):
    @abstractmethod
    def __init__(
        self,
        node_id: int,
        name: str,
        input_shapes: List[Tuple[int, ...]],
        output_shape: Tuple[int, ...],
        activation: Optional[Activation] = None,
    ) -> None:
        if len(input_shapes) < 1:
            raise ValueError("The input shape of the node is not filled")
        if () in input_shapes:
            raise ValueError("One of the input_shapes has dimension 0")
        if () == output_shape:
            raise ValueError("The output_shapes has dimension 0")
        self.id: int = node_id
        self.name: str = name
        self.input_shapes: List[Dimension] = list(map(Dimension, input_shapes))
        self.output_shape: Dimension = Dimension(output_shape)
        self.parents: List["Node"] = []
        self.activation: Activation = activation

    @abstractmethod
    def set_parents(self, nodes: Union["Node", List["Node"]]) -> None:
        pass
