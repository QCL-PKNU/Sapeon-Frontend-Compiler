from typing import List, Tuple
import pytest

from pyspear import nn

NODE_ID: int = 0
NODE_NAME: str = "reorg"
INPUT_SHAPES: List[Tuple] = [(1, 64, 112, 112)]
OUTPUT_SHAPE: Tuple = (1, 64, 56, 56)
ACTIVATION: nn.Activation = nn.Identity()


def test_init():
    reorg = nn.Reorg(
        node_id=NODE_ID,
        name=NODE_NAME,
        input_shapes=INPUT_SHAPES,
        output_shape=OUTPUT_SHAPE,
        stride=2,
        activation=ACTIVATION,
    )

    assert reorg.id == NODE_ID
    assert reorg.name == NODE_NAME
    assert reorg.stride == (2, 2)
    assert len(reorg.input_shapes) == 1
    assert reorg.input_shapes[0].dimension == INPUT_SHAPES[0]
    assert reorg.output_shape.dimension == OUTPUT_SHAPE


def test_no_activation():
    reorg = nn.Reorg(
        node_id=NODE_ID,
        name=NODE_NAME,
        input_shapes=INPUT_SHAPES,
        output_shape=OUTPUT_SHAPE,
        stride=2,
    )

    assert reorg.activation is None


def test_invalid_input_shapes():
    wrong_input_shapes = [(1, 64, 112, 112), (1, 1, 1, 1)]
    with pytest.raises(ValueError):
        nn.Reorg(
            node_id=NODE_ID,
            name=NODE_NAME,
            input_shapes=wrong_input_shapes,
            output_shape=OUTPUT_SHAPE,
            stride=2,
        )


def test_set_parents():
    reorg = nn.Reorg(
        node_id=NODE_ID,
        name=NODE_NAME,
        input_shapes=INPUT_SHAPES,
        output_shape=OUTPUT_SHAPE,
        stride=2,
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

    reorg.set_parents(conv1)

    assert reorg.parents == [conv1]
    with pytest.raises(ValueError):
        reorg.set_parents([])
    assert reorg.parents == [conv1]

    with pytest.raises(ValueError):
        reorg.set_parents([conv1, conv1])
    assert reorg.parents == [conv1]

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
        reorg.set_parents(connected)
    assert reorg.parents == [conv1]
