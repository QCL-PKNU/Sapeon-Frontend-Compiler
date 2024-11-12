from typing import Optional, List, Union

import pb2.spear.proto.e8e8_pb2 as proto
from pyspear import nn
from pyspear.nn.node import Node
from pyspear.nn.tensor import DataType, Tensor, DimFormat
from pyspear.nn.dimension import Dimension
from pyspear.nn.activation import Activation, ActivationMode


def create_default_layer(
    node: Node, types: Union[str, List[str]]
) -> proto.SPLayer:
    layer = proto.SPLayer()
    layer.id = node.id
    layer.name = node.name
    for x in node.parents:
        layer.preds.append(x.id)

    if isinstance(types, str):
        types = [
            types,
        ]

    for type_name in types:
        layer.type.append(type_name)

    if len(node.input_shapes) == 0:
        raise ValueError(f"{type(node).__name__}'s input_layers are not setted")
    inputs = list(map(create_shape_tensor, node.input_shapes))
    layer.input.extend(inputs)

    if node.output_shape is None:
        raise ValueError(f"{type(node).__name__}'s output_layer is not setted")
    output = create_shape_tensor(node.output_shape)
    layer.output.CopyFrom(output)

    if node.activation:
        layer.activation = get_activation_mode(node.activation)
        layer.type.append("activation")
        attributes = get_activation_attributes(node.activation)
        if len(attributes) > 0:
            layer.attributes.extend(attributes)

    return layer


def create_matmul_layer(
    node: Node, types: Union[str, List[str]]
) -> proto.SPLayer:
    layer = create_default_layer(node, types)

    weights = create_tensor(node.weights)
    layer.filter.CopyFrom(weights)
    bias = create_tensor(node.bias)
    layer.bias.CopyFrom(bias)

    convdesc = create_convdesc(node)
    layer.convdesc.CopyFrom(convdesc)

    return layer


def create_convdesc(node: nn.Node) -> proto.SPLayer.SPConvolutionDesc:
    convdesc = proto.SPLayer.SPConvolutionDesc()
    if hasattr(node, "padding"):
        convdesc.padding.extend(node.padding)
    else:
        convdesc.padding.extend((0, 0, 0, 0))

    if hasattr(node, "stride"):
        convdesc.stride.extend(node.stride)
    else:
        convdesc.stride.extend((1, 1))

    if hasattr(node, "dilation"):
        convdesc.dilation.extend(node.dilation)
    else:
        convdesc.dilation.extend((1, 1))

    if hasattr(node, "group"):
        convdesc.groups = node.group
    else:
        convdesc.groups = 1

    return convdesc


def create_samplingdesc(
    node: nn.Node, sampling_mode: proto.SPLayer.SPSamplingMode
) -> proto.SPLayer.SPSamplingDesc:
    samplingdesc = proto.SPLayer.SPSamplingDesc()
    samplingdesc.mode = sampling_mode
    if hasattr(node, "padding"):
        samplingdesc.padding.extend(node.padding)
    else:
        samplingdesc.padding.extend((0, 0, 0, 0))

    if hasattr(node, "stride"):
        samplingdesc.stride.extend(node.stride)
    else:
        samplingdesc.stride.extend((0, 0))

    if hasattr(node, "window"):
        samplingdesc.window.extend(node.window)

    return samplingdesc


def create_ewadddesc(node: nn.Ewadd) -> proto.SPLayer.SPEWAddDesc:
    ewadddesc = proto.SPLayer.SPEWAddDesc()

    # ewadd receives only two tensors
    if hasattr(node, "scale"):
        ewadddesc.scale.extend(node.scale)
    else:
        ewadddesc.scale.extend((1.0, 1.0))

    return ewadddesc


def create_ewmuldesc(node: nn.Ewmul) -> proto.SPLayer.SPEWMulDesc:
    ewmuldesc = proto.SPLayer.SPEWMulDesc()

    # ewmul receives only two tensors
    if hasattr(node, "scale"):
        ewmuldesc.scale.extend(node.scale)
    else:
        ewmuldesc.scale.extend((1.0, 1.0))

    return ewmuldesc


def create_shape_tensor(dim: Dimension) -> proto.SPLayer.SPTensor:
    sptensor = proto.SPLayer.SPTensor()
    sptensor.dtype = get_proto_dtype(None)
    sptensor.format = get_proto_dimformat(None)
    sptensor.size = dim.size

    # spear.proto.e8e8's dimension order is w, h, c, n if format is nchw
    for d in reversed(dim.dimension):
        sptensor.dims.append(d)

    return sptensor


def create_tensor(tensor: Tensor) -> proto.SPLayer.SPTensor:
    sptensor = proto.SPLayer.SPTensor()
    sptensor.dtype = get_proto_dtype(tensor.datatype)
    sptensor.format = get_proto_dimformat(tensor.dim_format)
    sptensor.size = tensor.size

    # spear.proto.e8e8's dimension order is w, h, c, n if format is nchw
    for dim in reversed(tensor.dimension):
        sptensor.dims.append(dim)

    if tensor.data is None:
        return sptensor

    if tensor.datatype in (
        None,
        DataType.float,
        DataType.double,
        DataType.float16,
    ):
        sptensor.fval.extend(tensor.data)
    elif tensor.datatype in (
        DataType.uint8,
        DataType.sint8,
        DataType.sint16,
    ):
        sptensor.bval.extend(tensor.data)
    else:
        raise ValueError(
            "Loaded values' datatype in tensor not supported for "
            "spear.proto.e8e8"
        )
    return sptensor


def get_proto_dtype(
    dtype: Optional[DataType] = None,
) -> proto.SPLayer.SPDataType:
    # default is SP_DATA_FLOAT
    if dtype is None or dtype == DataType.float:
        return proto.SPLayer.SPDataType.SP_DATA_FLOAT
    if dtype == DataType.double:
        return proto.SPLayer.SPDataType.SP_DATA_DOUBLE
    if dtype == DataType.float16:
        return proto.SPLayer.SPDataType.SP_DATA_HALF
    if dtype == DataType.uint8:
        return proto.SPLayer.SPDataType.SP_DATA_UINT8
    if dtype == DataType.sint8:
        return proto.SPLayer.SPDataType.SP_DATA_SINT8
    if dtype == DataType.sint16:
        return proto.SPLayer.SPDataType.SP_DATA_SINT16
    raise ValueError("Not supported datatype for spear.proto.e8e8")


def get_proto_dimformat(
    dimfmt: Optional[DimFormat] = None,
) -> proto.SPLayer.SPTensorFormat:
    # default is SP_FORMAT_NCHW
    if dimfmt is None or dimfmt == DimFormat.NCHW:
        return proto.SPLayer.SPTensorFormat.SP_FORMAT_NCHW
    if dimfmt == DimFormat.NHWC:
        return proto.SPLayer.SPTensorFormat.SP_FORMAT_NHWC
    if dimfmt == DimFormat.NWHC:
        return proto.SPLayer.SPTensorFormat.SP_FORMAT_NWHC
    if dimfmt == DimFormat.VECTOR:
        return proto.SPLayer.SPTensorFormat.SP_FORMAT_VECTOR
    raise ValueError("Not supported dimension format for spear.proto.e8e8")


def get_activation_mode(
    activation: Activation,
) -> proto.SPLayer.SPActivationMode:
    if activation is None:
        return proto.SPLayer.SPActivationMode.SP_ACTIVATION_IDENTITY

    mode: ActivationMode = None
    try:
        mode = activation.activation_mode()
    except AttributeError as exc:
        raise ValueError(
            "Not supported activation mode for spear.proto.e8e8"
        ) from exc

    if mode is None or mode == ActivationMode.Identity:
        return proto.SPLayer.SPActivationMode.SP_ACTIVATION_IDENTITY
    if mode == ActivationMode.Sigmoid:
        return proto.SPLayer.SPActivationMode.SP_ACTIVATION_SIGMOID
    if mode == ActivationMode.ReLU:
        return proto.SPLayer.SPActivationMode.SP_ACTIVATION_RELU
    if mode == ActivationMode.LeakyReLU:
        return proto.SPLayer.SPActivationMode.SP_ACTIVATION_LEAKY_RELU
    if mode == ActivationMode.PReLU:
        return proto.SPLayer.SPActivationMode.SP_ACTIVATION_PRELU
    if mode == ActivationMode.Tanh:
        return proto.SPLayer.SPActivationMode.SP_ACTIVATION_TANH
    if mode == ActivationMode.Mish:
        return proto.SPLayer.SPActivationMode.SP_ACTIVATION_MISH
    if mode == ActivationMode.ReLU6:
        return proto.SPLayer.SPActivationMode.SP_ACTIVATION_RELU6
    if mode == ActivationMode.Swish:
        return proto.SPLayer.SPActivationMode.SP_ACTIVATION_SWISH
    raise ValueError("Not supported activation mode for spear.proto.e8e8")


def get_activation_attributes(
    activation: Activation,
) -> List[proto.SPLayer.Attribute]:
    attributes = []
    if isinstance(activation, nn.LeakyReLU):
        attribute = proto.SPLayer.Attribute()
        attribute.name = "leaky_slope"
        attribute.type = proto.SPLayer.Attribute.AttributeType.FLOAT
        attribute.f = activation.leaky_slope
        attributes.append(attribute)
    elif isinstance(activation, nn.PReLU):
        attribute = proto.SPLayer.Attribute()
        attribute.name = "neg_slope"
        if isinstance(activation.neg_slope, float):
            attribute.type = proto.SPLayer.Attribute.AttributeType.FLOAT
            attribute.f = activation.neg_slope
        elif isinstance(activation.neg_slope, List[str]):
            attribute.type = proto.SPLayer.Attribute.AttributeType.FLOATS
            attribute.floats.extend(activation.neg_slope)
        attributes.append(attribute)

    return attributes
