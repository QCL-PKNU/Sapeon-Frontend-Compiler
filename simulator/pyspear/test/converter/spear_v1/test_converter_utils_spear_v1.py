from typing import List, Tuple
from functools import reduce

import pytest
import numpy as np

import pb2.spear.proto.e8e8_pb2 as proto
from pyspear import nn
from pyspear.nn.dimension import Dimension
from pyspear.nn.tensor import Tensor, DataType, DimFormat
import pyspear.converter.spear_v1.converter_utils_spear_v1 as util

NODE_ID: int = 0
NODE_NAME: str = "maxpool"
INPUT_SHAPES: List[Tuple] = [(1, 64, 112, 112)]
OUTPUT_SHAPE: Tuple = (1, 64, 56, 56)
ACTIVATION: nn.Activation = nn.Identity()


def test_create_default_layer():
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
    splayer = util.create_default_layer(node=node, types="maxpool")

    assert splayer.id == NODE_ID
    assert splayer.name == NODE_NAME
    assert splayer.type[0] == "maxpool"
    for sp, ref in zip(splayer.input[0].dims, reversed(INPUT_SHAPES[0])):
        assert sp == ref

    assert len(splayer.input) > 0
    assert splayer.input[0] is not None
    assert splayer.output is not None

    assert (
        splayer.activation
        == proto.SPLayer.SPActivationMode.SP_ACTIVATION_IDENTITY
    )


def test_create_matmul_layer():
    padding = 3
    stride = 2
    dilation = 1
    group = 1
    kernel_size = 7
    conv = nn.Conv(
        node_id=0,
        name="conv",
        input_shapes=[(1, 3, 224, 224)],
        output_shape=(1, 64, 112, 112),
        in_channels=3,
        out_channels=64,
        kernel_size=kernel_size,
        weight_dtype="float32",
        bias_dtype="float32",
        padding=padding,
        stride=stride,
        dilation=dilation,
        group=group,
        activation=nn.ReLU(),
    )

    matmul_layer = util.create_matmul_layer(conv, "convolution")
    assert matmul_layer.filter is not None
    assert matmul_layer.bias is not None
    assert matmul_layer.convdesc is not None


def test_create_shape_tensor():
    dim = Dimension(dimension=(1, 2, 3, 4))
    sptensor = util.create_shape_tensor(dim)
    assert sptensor.dtype == proto.SPLayer.SPDataType.SP_DATA_FLOAT
    assert sptensor.format == proto.SPLayer.SPTensorFormat.SP_FORMAT_NCHW
    assert sptensor.size == reduce(lambda x, y: x * y, dim.dimension)
    for sp, ref in zip(sptensor.dims, reversed(dim.dimension)):
        assert sp == ref


def test_create_tensor():
    tensor = Tensor((1, 2, 3, 1))

    random_array = np.random.rand(1, 2, 3, 1)
    tensor.load_ndarray(random_array)

    sptensor = util.create_tensor(tensor=tensor)
    assert sptensor.dtype == proto.SPLayer.SPDataType.SP_DATA_FLOAT
    assert sptensor.format == proto.SPLayer.SPTensorFormat.SP_FORMAT_NCHW
    assert sptensor.size == reduce(lambda x, y: x * y, tensor.dimension)
    for sp, ref in zip(sptensor.dims, reversed(tensor.dimension)):
        assert sp == ref

    for sp, ref in zip(sptensor.fval, tensor.data):
        assert sp == pytest.approx(ref)


def test_get_proto_dtype():
    dtype = util.get_proto_dtype(DataType.float)
    assert dtype == proto.SPLayer.SPDataType.SP_DATA_FLOAT

    dtype = util.get_proto_dtype(DataType.double)
    assert dtype == proto.SPLayer.SPDataType.SP_DATA_DOUBLE

    dtype = util.get_proto_dtype()
    assert dtype == proto.SPLayer.SPDataType.SP_DATA_FLOAT

    with pytest.raises(ValueError):
        dtype = util.get_proto_dtype("wrong_input")


def test_get_proto_dimformat():
    dtype = util.get_proto_dimformat(DimFormat.NCHW)
    assert dtype == proto.SPLayer.SPTensorFormat.SP_FORMAT_NCHW

    dtype = util.get_proto_dimformat(DimFormat.VECTOR)
    assert dtype == proto.SPLayer.SPTensorFormat.SP_FORMAT_VECTOR

    dtype = util.get_proto_dimformat()
    assert dtype == proto.SPLayer.SPTensorFormat.SP_FORMAT_NCHW

    with pytest.raises(ValueError):
        dtype = util.get_proto_dimformat("wrong_input")


def test_get_activation_mode():
    spactmode = util.get_activation_mode(nn.Identity())
    assert spactmode == proto.SPLayer.SPActivationMode.SP_ACTIVATION_IDENTITY

    spactmode = util.get_activation_mode(None)
    assert spactmode == proto.SPLayer.SPActivationMode.SP_ACTIVATION_IDENTITY

    spactmode = util.get_activation_mode(nn.ReLU())
    assert spactmode == proto.SPLayer.SPActivationMode.SP_ACTIVATION_RELU

    with pytest.raises(ValueError):
        spactmode = util.get_activation_mode("wrong_input")


def test_create_ewmuldesc():
    node = nn.Ewmul(
        node_id=0,
        name="ewmul",
        input_shapes=[(1, 2, 3, 4), (1, 2, 3, 4)],
        output_shape=(1, 2, 3, 4),
        scale=2.0,
    )
    ewmuldesc = util.create_ewmuldesc(node)
    for scale in ewmuldesc.scale:
        assert scale == 2.0

    node = nn.Ewmul(
        node_id=0,
        name="ewmul",
        input_shapes=[(1, 2, 3, 4), (1, 2, 3, 4)],
        output_shape=(1, 2, 3, 4),
    )
    ewmuldesc = util.create_ewmuldesc(node)
    for scale in ewmuldesc.scale:
        assert scale == 1.0

    node = nn.Ewmul(
        node_id=0,
        name="ewmul",
        input_shapes=[(1, 2, 3, 4), (1, 2, 3, 4)],
        output_shape=(1, 2, 3, 4),
        scale=(1.0, 2.0),
    )
    ewmuldesc = util.create_ewmuldesc(node)
    for ewscale, origscale in zip(ewmuldesc.scale, (1.0, 2.0)):
        assert ewscale == origscale


def test_create_ewaddesc():
    node = nn.Ewadd(
        node_id=0,
        name="ewadd",
        input_shapes=[(1, 2, 3, 4), (1, 2, 3, 4)],
        output_shape=(1, 2, 3, 4),
        scale=2.0,
    )
    ewadddesc = util.create_ewadddesc(node)
    for scale in ewadddesc.scale:
        assert scale == 2.0

    node = nn.Ewadd(
        node_id=0,
        name="ewadd",
        input_shapes=[(1, 2, 3, 4), (1, 2, 3, 4)],
        output_shape=(1, 2, 3, 4),
    )
    ewadddesc = util.create_ewadddesc(node)
    for scale in ewadddesc.scale:
        assert scale == 1.0

    node = nn.Ewadd(
        node_id=0,
        name="ewadd",
        input_shapes=[(1, 2, 3, 4), (1, 2, 3, 4)],
        output_shape=(1, 2, 3, 4),
        scale=(1.0, 2.0),
    )
    ewadddesc = util.create_ewadddesc(node)
    for ewscale, origscale in zip(ewadddesc.scale, (1.0, 2.0)):
        assert ewscale == origscale


def test_create_samplingdesc():
    padding = 1
    stride = 2
    window = 3
    node = nn.Maxpool(
        node_id=NODE_ID,
        name=NODE_NAME,
        input_shapes=INPUT_SHAPES,
        output_shape=OUTPUT_SHAPE,
        padding=padding,
        window=window,
        stride=stride,
        activation=ACTIVATION,
    )

    samplingdesc = util.create_samplingdesc(
        node, proto.SPLayer.SPSamplingMode.SP_POOLING_MAX
    )

    for pad in samplingdesc.padding:
        assert pad == padding

    for st in samplingdesc.stride:
        assert st == stride

    for win in samplingdesc.window:
        assert win == window


def test_create_convdesc():
    padding = 3
    stride = 2
    dilation = 1
    group = 1
    kernel_size = 7
    conv = nn.Conv(
        node_id=0,
        name="conv",
        input_shapes=[(1, 3, 224, 224)],
        output_shape=(1, 64, 112, 112),
        in_channels=3,
        out_channels=64,
        kernel_size=kernel_size,
        weight_dtype="float32",
        bias_dtype="float32",
        padding=padding,
        stride=stride,
        dilation=dilation,
        group=group,
        activation=nn.ReLU(),
    )

    convdesc = util.create_convdesc(conv)

    for pad in convdesc.padding:
        assert pad == padding

    for st in convdesc.stride:
        assert st == stride

    for dil in convdesc.dilation:
        assert dil == dilation

    assert convdesc.groups == group
