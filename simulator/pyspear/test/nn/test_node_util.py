from typing import Callable, Tuple
import pytest

from pyspear import nn
import pyspear.nn.node_util as util


def create_nodes() -> Tuple[nn.Node]:
    conv0 = nn.Conv(
        node_id=0,
        name="/relu/Relu_output_0",
        input_shapes=[(1, 3, 224, 224)],
        output_shape=(1, 64, 112, 112),
        in_channels=3,
        out_channels=64,
        kernel_size=7,
        weight_dtype="float32",
        bias_dtype="float32",
        padding=3,
        stride=2,
        activation=nn.ReLU(),
    )

    maxpool1 = nn.Maxpool(
        node_id=1,
        name="/maxpool/MaxPool_output_0",
        input_shapes=[(1, 64, 112, 112)],
        output_shape=(1, 64, 56, 56),
        padding=1,
        window=3,
        stride=2,
    )

    conv2 = nn.Conv(
        node_id=2,
        name="/layer1/layer1.0/relu/Relu_output_0",
        input_shapes=[(1, 64, 56, 56)],
        output_shape=(1, 64, 56, 56),
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        padding=1,
        stride=1,
        activation=nn.ReLU(),
    )

    conv3 = nn.Conv(
        node_id=3,
        name="/layer1/layer1.0/conv2/Conv_output_0",
        input_shapes=[(1, 64, 56, 56)],
        output_shape=(1, 64, 56, 56),
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        padding=1,
        stride=1,
    )

    ewadd4 = nn.Ewadd(
        node_id=4,
        name="/layer1/layer1.0/relu_1/Relu_output_0",
        input_shapes=[(1, 64, 56, 56), (1, 64, 56, 56)],
        output_shape=(1, 64, 56, 56),
        activation=nn.ReLU(),
    )

    return (conv0, maxpool1, conv2, conv3, ewadd4)


def convert_to_2dim_function_test(func: Callable):
    scalar = 2
    twodim = (1, 1)
    twolist = [1, 1]
    fourdim = (1, 1, 1, 1)
    assert func(scalar) == (2, 2)
    assert func(twodim) == twodim

    with pytest.raises(ValueError):
        func(fourdim)
    with pytest.raises(ValueError):
        func(twolist)
    with pytest.raises(ValueError):
        func("hello")


def test_convert_kernel_size():
    convert_to_2dim_function_test(util.convert_kernel_size)


def test_convert_stride():
    convert_to_2dim_function_test(util.convert_stride)


def test_convert_window():
    convert_to_2dim_function_test(util.convert_window)


def test_convert_dilation():
    convert_to_2dim_function_test(util.convert_dilation)


def test_convert_scale():
    int_scalar = 2
    scalar = 2.0
    int_twodim = (1, 1)
    twodim = (1.0, 1.0)
    twolist = [1.0, 1.0]
    fourdim = (1.0, 1.0, 1.0, 1.0)
    assert util.convert_scale(scalar) == (2, 2)
    assert util.convert_scale(twodim) == twodim
    assert util.convert_scale(int_twodim) == int_twodim

    with pytest.raises(ValueError):
        util.convert_scale(int_scalar)
    with pytest.raises(ValueError):
        util.convert_scale(fourdim)
    with pytest.raises(ValueError):
        util.convert_scale(twolist)
    with pytest.raises(ValueError):
        util.convert_scale("hello")


def test_convert_padding():
    scalar = 2
    # height, width
    twodim = (1, 2)
    twolist = [1, 2]
    threedim = (1, 2, 3)
    # top, bottom, left, right
    fourdim = (1, 2, 3, 4)
    assert util.convert_padding(scalar) == (2, 2, 2, 2)
    assert util.convert_padding(twodim) == (1, 1, 2, 2)
    assert util.convert_padding(fourdim) == (1, 2, 3, 4)

    with pytest.raises(ValueError):
        util.convert_padding(threedim)
    with pytest.raises(ValueError):
        util.convert_padding(twolist)
    with pytest.raises(ValueError):
        util.convert_padding("hello")


def test_convert_nodes():
    nodes = create_nodes()

    assert util.convert_nodes(nodes) == nodes
    assert util.convert_nodes(nodes[0]) == [nodes[0]]


def test_assert_parent_nodes_inputs():
    conv0, maxpool1, conv2, conv3, ewadd4 = create_nodes()

    util.assert_parent_nodes_inputs(child=maxpool1, parents=[conv0])
    with pytest.raises(ValueError):
        util.assert_parent_nodes_inputs(child=maxpool1, parents=[conv2])
    with pytest.raises(ValueError):
        util.assert_parent_nodes_inputs(child=maxpool1, parents=[conv0, conv2])

    util.assert_parent_nodes_inputs(child=ewadd4, parents=[conv2, conv3])
    with pytest.raises(ValueError):
        util.assert_parent_nodes_inputs(child=ewadd4, parents=[conv2])
    with pytest.raises(ValueError):
        util.assert_parent_nodes_inputs(child=ewadd4, parents=[conv2, conv0])
