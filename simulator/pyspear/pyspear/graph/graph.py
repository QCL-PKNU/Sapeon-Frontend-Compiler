from typing import List, Any
from abc import ABC, abstractmethod

from pyspear.nn.node import Node


class Graph(ABC):
    @abstractmethod
    def set_input_nodes(self, nodes: List[Node]):
        pass

    @abstractmethod
    def set_output_nodes(self, nodes: List[Node]):
        pass

    @abstractmethod
    def serialize(self) -> Any:
        pass

    def export(self, filename: str):
        serialized_graph = self.serialize()
        with open(filename, "wb") as f:
            f.write(serialized_graph.SerializeToString())
