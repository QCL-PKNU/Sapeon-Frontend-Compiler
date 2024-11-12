import pb2.spear.proto.e8e8_pb2 as proto
from pyspear.nn.connected import Connected
from pyspear.converter.node_converter import NodeConverter
from .converter_utils_spear_v1 import create_matmul_layer


class ConnectedConverterSpearV1(NodeConverter):
    def to_proto(self, node: Connected) -> proto.SPLayer:
        layer = create_matmul_layer(node, "connected")

        # input_threshold
        # output_threshold
        # filter_threshold

        return layer
