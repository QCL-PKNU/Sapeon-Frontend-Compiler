import pb2.spear.proto.e8e8_pb2 as proto
from pyspear.nn.conv import Conv
from pyspear.converter.node_converter import NodeConverter
from .converter_utils_spear_v1 import create_matmul_layer


class ConvConverterSpearV1(NodeConverter):
    def to_proto(self, node: Conv) -> proto.SPLayer:
        layer_name = "convolution"
        if node.group > 1:
            layer_name = "groupconv"
        layer = create_matmul_layer(node, layer_name)

        # input_threshold
        # output_threshold
        # filter_threshold

        return layer
