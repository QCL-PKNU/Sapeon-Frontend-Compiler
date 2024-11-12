import pb2.spear.proto.e8e8_pb2 as proto
from pyspear.nn.maxpool import Maxpool
from pyspear.converter.node_converter import NodeConverter
from .converter_utils_spear_v1 import create_default_layer, create_samplingdesc


class MaxpoolConverterSpearV1(NodeConverter):
    def to_proto(self, node: Maxpool) -> proto.SPLayer:
        layer = create_default_layer(node, "maxpool")

        samplingdesc = create_samplingdesc(
            node,
            sampling_mode=proto.SPLayer.SPSamplingMode.SP_POOLING_MAX,
        )
        layer.samplingdesc.CopyFrom(samplingdesc)

        # input_threshold
        # output_threshold

        return layer
