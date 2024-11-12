import pb2.spear.proto.e8e8_pb2 as proto
from pyspear.nn.ewadd import Ewadd
from pyspear.converter.node_converter import NodeConverter
from .converter_utils_spear_v1 import (
    create_default_layer,
    create_ewadddesc,
)


class EwaddConverterSpearV1(NodeConverter):
    def to_proto(self, node: Ewadd) -> proto.SPLayer:
        if len(node.input_shapes) != 2:
            raise ValueError(
                f"The input layers of the {type(node).__name__} are not set to"
                " 2"
            )
        layer = create_default_layer(node, "ewadd")

        ewadddesc = create_ewadddesc(node)
        layer.ewadddesc.CopyFrom(ewadddesc)

        # input_threshold
        # output_threshold

        return layer
