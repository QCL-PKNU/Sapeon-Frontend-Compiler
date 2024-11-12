from typing import Any
from abc import ABC, abstractmethod

from pyspear.nn.node import Node


class NodeConverter(ABC):
    @abstractmethod
    def to_proto(self, node: Node) -> Any:
        pass
