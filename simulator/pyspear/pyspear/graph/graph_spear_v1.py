from typing import Dict, Type, List, Any
from collections import deque

import pb2.spear.proto.e8e8_pb2 as proto
from pyspear.graph.graph import Graph
from pyspear import nn
from pyspear.converter.node_converter import NodeConverter
import pyspear.converter.spear_v1 as converter


class GraphSpearV1(Graph):
    def __init__(self) -> None:
        self.layers: Dict[int, nn.Node] = {}
        self.input_layers: Dict[int, nn.Node] = {}
        self.output_layers: Dict[int, nn.Node] = {}
        self.converter: Dict[Type[nn.Node], NodeConverter] = {
            nn.Connected: converter.ConnectedConverterSpearV1(),
            nn.Conv: converter.ConvConverterSpearV1(),
            nn.Ewadd: converter.EwaddConverterSpearV1(),
            nn.Ewmul: converter.EwmulConverterSpearV1(),
            nn.Gavgpool: converter.GavgpoolConverterSpearV1(),
            nn.Lavgpool: converter.LavgpoolConverterSpearV1(),
            nn.Maxpool: converter.MaxpoolConverterSpearV1(),
            nn.Pixelshuffle: converter.PixelshuffleConverterSpearV1(),
            nn.Reorg: converter.ReorgConverterSpearV1(),
            nn.Route: converter.RouteConverterSpearV1(),
            nn.Upsample: converter.UpsampleConverterSpearV1(),
        }

    def set_input_nodes(self, nodes: List[nn.Node]) -> None:
        self.input_layers.clear()
        for input_node in nodes:
            self.input_layers[input_node.id] = input_node

    def set_output_nodes(self, nodes: List[nn.Node]) -> None:
        for output_node in nodes:
            self.output_layers[output_node.id] = output_node
            if output_node.id not in self.layers:
                self.layers[output_node.id] = output_node
            next_nodes = deque(output_node.parents)
            while len(next_nodes) > 0:
                node = next_nodes.popleft()
                if node.id not in self.layers:
                    self.layers[node.id] = node
                next_nodes.extend(node.parents)
        self.layers = {key: self.layers[key] for key in sorted(self.layers)}

    def serialize(self) -> Any:
        spgraph = proto.SPGraph()
        for node_id in self.input_layers:
            spgraph.input_layers.append(node_id)

        for node_id in self.output_layers:
            spgraph.output_layers.append(node_id)

        for node in self.layers.values():
            layer = self.convert_node_to_layer(node)
            spgraph.layer.append(layer)

        for layer in spgraph.layer:
            for pred in layer.preds:
                spgraph.layer[pred].succs.append(layer.id)
        return spgraph

    def convert_node_to_layer(self, node: nn.Node) -> proto.SPLayer:
        try:
            e8e8_converter = self.converter[type(node)]
            return e8e8_converter.to_proto(node)
        except KeyError as e:
            raise KeyError(
                f"{type(node).__name__}'s E8E8Converter is not implemented"
            ) from e
