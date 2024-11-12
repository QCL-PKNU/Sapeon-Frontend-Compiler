from typing import List, Tuple
import pytest

import pb2.spear.proto.e8e8_pb2 as proto
from pyspear import nn
from pyspear.converter.spear_v1 import MaxpoolConverterSpearV1

NODE_ID: int = 0
NODE_NAME: str = "maxpool1"
INPUT_SHAPES: List[Tuple] = [(1, 64, 112, 112)]
OUTPUT_SHAPE: Tuple = (1, 64, 56, 56)
ACTIVATION: nn.Activation = nn.Identity()


def test_to_proto():
    node = nn.Maxpool(
        node_id=NODE_ID,
        name=NODE_NAME,
        input_shapes=INPUT_SHAPES,
        output_shape=OUTPUT_SHAPE,
        padding=1,
        window=3,
        stride=2,
        activation=ACTIVATION,
    )

    splayer = MaxpoolConverterSpearV1().to_proto(node)

    assert splayer.id == NODE_ID
    assert splayer.name == NODE_NAME
    assert splayer.type[0] == "maxpool"
    for sp, ref in zip(splayer.input[0].dims, reversed(INPUT_SHAPES[0])):
        assert sp == ref

    assert len(splayer.input) == 1
    assert splayer.input[0] is not None
    assert splayer.output is not None

    assert (
        splayer.activation
        == proto.SPLayer.SPActivationMode.SP_ACTIVATION_IDENTITY
    )

    assert splayer.samplingdesc is not None
    assert (
        splayer.samplingdesc.mode == proto.SPLayer.SPSamplingMode.SP_POOLING_MAX
    )
