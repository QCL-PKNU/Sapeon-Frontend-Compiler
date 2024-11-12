from typing import List, Tuple
import pytest

from pyspear import nn

NODE_ID: int = 0
NODE_NAME: str = "maxpool1"
INPUT_SHAPES: List[Tuple] = [(1, 64, 112, 112)]
OUTPUT_SHAPE: Tuple = (1, 64, 56, 56)
ACTIVATION: nn.Activation = nn.Identity()


def test_init():
    maxpool = nn.Maxpool(
        node_id=NODE_ID,
        name=NODE_NAME,
        input_shapes=INPUT_SHAPES,
        output_shape=OUTPUT_SHAPE,
        padding=1,
        window=3,
        stride=2,
        activation=ACTIVATION,
    )

    assert maxpool.id == NODE_ID
    assert maxpool.name == NODE_NAME
    assert maxpool.padding == (1, 1, 1, 1)
    assert maxpool.window == (3, 3)
    assert maxpool.stride == (2, 2)
    assert len(maxpool.input_shapes) == 1
    assert maxpool.input_shapes[0].dimension == INPUT_SHAPES[0]
    assert maxpool.output_shape.dimension == OUTPUT_SHAPE
    assert maxpool.activation == ACTIVATION


def test_no_padding():
    maxpool = nn.Maxpool(
        node_id=NODE_ID,
        name=NODE_NAME,
        input_shapes=INPUT_SHAPES,
        output_shape=OUTPUT_SHAPE,
        window=3,
        stride=2,
        activation=ACTIVATION,
    )

    assert maxpool.padding == (0, 0, 0, 0)


def test_no_stride():
    maxpool = nn.Maxpool(
        node_id=NODE_ID,
        name=NODE_NAME,
        input_shapes=INPUT_SHAPES,
        output_shape=OUTPUT_SHAPE,
        window=3,
        padding=1,
        activation=ACTIVATION,
    )

    assert maxpool.stride == maxpool.window


def test_no_activation():
    maxpool = nn.Maxpool(
        node_id=NODE_ID,
        name=NODE_NAME,
        input_shapes=INPUT_SHAPES,
        output_shape=OUTPUT_SHAPE,
        padding=1,
        window=3,
        stride=2,
    )

    assert maxpool.activation is None


def test_invalid_input_shapes():
    wrong_input_shapes = [(1, 64, 112, 112), (1, 1, 1, 1)]
    with pytest.raises(ValueError):
        nn.Maxpool(
            node_id=NODE_ID,
            name=NODE_NAME,
            input_shapes=wrong_input_shapes,
            output_shape=OUTPUT_SHAPE,
            padding=1,
            window=3,
            stride=2,
            activation=ACTIVATION,
        )


def test_set_parents():
    maxpool = nn.Maxpool(
        node_id=NODE_ID,
        name=NODE_NAME,
        input_shapes=INPUT_SHAPES,
        output_shape=OUTPUT_SHAPE,
        window=3,
    )

    conv1 = nn.Conv(
        node_id=0,
        name="conv1",
        input_shapes=[(1, 3, 224, 224)],
        output_shape=(1, 64, 112, 112),
        in_channels=3,
        out_channels=64,
        kernel_size=7,
        weight_dtype="float32",
        bias_dtype="float32",
        padding=3,
        stride=2,
        dilation=1,
        group=1,
        activation=nn.LeakyReLU(0.1),
    )

    maxpool.set_parents(conv1)

    assert maxpool.parents == [conv1]
    with pytest.raises(ValueError):
        maxpool.set_parents([])
    assert maxpool.parents == [conv1]

    with pytest.raises(ValueError):
        maxpool.set_parents([conv1, conv1])
    assert maxpool.parents == [conv1]

    connected = nn.Connected(
        node_id=2,
        name="connected1",
        input_shapes=[(1, 2048, 1, 1)],
        output_shape=(1, 1001, 1, 1),
        in_channels=2048,
        out_channels=1001,
        weight_dtype="float32",
        bias_dtype="float32",
    )

    with pytest.raises(ValueError):
        maxpool.set_parents(connected)
    assert maxpool.parents == [conv1]
