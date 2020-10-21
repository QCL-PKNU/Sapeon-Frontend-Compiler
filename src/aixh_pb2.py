# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: aixh.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='aixh.proto',
  package='aixh',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\naixh.proto\x12\x04\x61ixh\"\x80\x11\n\x08\x41IXLayer\x12\n\n\x02id\x18\x01 \x02(\x05\x12\x0c\n\x04name\x18\x02 \x01(\t\x12)\n\x04type\x18\x03 \x03(\x0e\x32\x1b.aixh.AIXLayer.AIXLayerType\x12\r\n\x05preds\x18\x04 \x03(\x05\x12\r\n\x05succs\x18\x05 \x03(\x05\x12\'\n\x05input\x18\x06 \x01(\x0b\x32\x18.aixh.AIXLayer.AIXTensor\x12(\n\x06output\x18\x07 \x01(\x0b\x32\x18.aixh.AIXLayer.AIXTensor\x12(\n\x06\x66ilter\x18\x08 \x01(\x0b\x32\x18.aixh.AIXLayer.AIXTensor\x12&\n\x04\x62ias\x18\t \x01(\x0b\x32\x18.aixh.AIXLayer.AIXTensor\x12\'\n\x05scale\x18\n \x01(\x0b\x32\x18.aixh.AIXLayer.AIXTensor\x12&\n\x04mean\x18\x0b \x01(\x0b\x32\x18.aixh.AIXLayer.AIXTensor\x12*\n\x08variance\x18\x0c \x01(\x0b\x32\x18.aixh.AIXLayer.AIXTensor\x12\x0f\n\x07\x65psilon\x18\x10 \x01(\x02\x12\x17\n\x0finput_threshold\x18\r \x01(\x02\x12\x18\n\x10output_threshold\x18\x0e \x01(\x02\x12\x18\n\x10\x66ilter_threshold\x18\x0f \x01(\x02\x12\x33\n\x08\x63onvdesc\x18\x14 \x02(\x0b\x32!.aixh.AIXLayer.AIXConvolutionDesc\x12\x34\n\x0csamplingdesc\x18\x15 \x01(\x0b\x32\x1e.aixh.AIXLayer.AIXSamplingDesc\x12.\n\tewadddesc\x18\x16 \x01(\x0b\x32\x1b.aixh.AIXLayer.AIXEWAddDesc\x12\x34\n\nactivation\x18\x1e \x01(\x0e\x32 .aixh.AIXLayer.AIXActivationMode\x1a\x82\x01\n\x12\x41IXConvolutionDesc\x12)\n\x05\x64type\x18\x01 \x02(\x0e\x32\x1a.aixh.AIXLayer.AIXDataType\x12\x0f\n\x07padding\x18\x02 \x03(\x05\x12\x0e\n\x06stride\x18\x03 \x03(\x05\x12\x10\n\x08\x64ilation\x18\x04 \x03(\x05\x12\x0e\n\x06groups\x18\x05 \x02(\x05\x1ap\n\x0f\x41IXSamplingDesc\x12,\n\x04mode\x18\x01 \x02(\x0e\x32\x1e.aixh.AIXLayer.AIXSamplingMode\x12\x0f\n\x07padding\x18\x02 \x03(\x05\x12\x0e\n\x06stride\x18\x03 \x03(\x05\x12\x0e\n\x06window\x18\x04 \x03(\x05\x1a\x1d\n\x0c\x41IXEWAddDesc\x12\r\n\x05scale\x18\x01 \x03(\x02\x1a\xaf\x01\n\tAIXTensor\x12)\n\x05\x64type\x18\x01 \x02(\x0e\x32\x1a.aixh.AIXLayer.AIXDataType\x12.\n\x06\x66ormat\x18\x02 \x02(\x0e\x32\x1e.aixh.AIXLayer.AIXTensorFormat\x12\x0c\n\x04\x64ims\x18\x03 \x03(\x05\x12\x0c\n\x04\x66val\x18\x04 \x03(\x02\x12\x10\n\x04\x62val\x18\x05 \x03(\x05\x42\x02\x10\x01\x12\x0c\n\x04size\x18\x06 \x01(\x05\x12\x0b\n\x03ptr\x18\x07 \x01(\x03\"\xba\x03\n\x0c\x41IXLayerType\x12\x19\n\x15\x41IX_LAYER_CONVOLUTION\x10\x00\x12\x17\n\x13\x41IX_LAYER_CONNECTED\x10\x01\x12\x15\n\x11\x41IX_LAYER_MAXPOOL\x10\x02\x12\x15\n\x11\x41IX_LAYER_AVGPOOL\x10\x03\x12\x15\n\x11\x41IX_LAYER_SOFTMAX\x10\x04\x12\x13\n\x0f\x41IX_LAYER_ROUTE\x10\x06\x12\x13\n\x0f\x41IX_LAYER_REORG\x10\x07\x12\x13\n\x0f\x41IX_LAYER_EWADD\x10\x08\x12\x16\n\x12\x41IX_LAYER_UPSAMPLE\x10\t\x12\x1a\n\x16\x41IX_LAYER_PIXELSHUFFLE\x10\n\x12\x18\n\x14\x41IX_LAYER_GROUP_CONV\x10\x0b\x12\x17\n\x13\x41IX_LAYER_SKIP_CONV\x10\x0c\x12\x18\n\x14\x41IX_LAYER_ACTIVATION\x10\r\x12\x17\n\x13\x41IX_LAYER_BATCHNORM\x10\x0e\x12\x15\n\x11\x41IX_LAYER_BIASADD\x10\x0f\x12\x14\n\x10\x41IX_LAYER_OUTPUT\x10\x10\x12\x13\n\x0f\x41IX_LAYER_INPUT\x10\x11\x12\x16\n\x12\x41IX_LAYER_WILDCARD\x10\x12\"\xb7\x01\n\x11\x41IXActivationMode\x12\x1a\n\x16\x41IX_ACTIVATION_SIGMOID\x10\x00\x12\x17\n\x13\x41IX_ACTIVATION_RELU\x10\x01\x12\x1d\n\x19\x41IX_ACTIVATION_LEAKY_RELU\x10\x02\x12\x18\n\x14\x41IX_ACTIVATION_PRELU\x10\x03\x12\x17\n\x13\x41IX_ACTIVATION_TANH\x10\x04\x12\x1b\n\x17\x41IX_ACTIVATION_IDENTITY\x10\x05\"\x8e\x01\n\x0f\x41IXSamplingMode\x12\x13\n\x0f\x41IX_POOLING_MAX\x10\x00\x12\x17\n\x13\x41IX_POOLING_AVERAGE\x10\x01\x12\x15\n\x11\x41IX_POOLING_REORG\x10\x02\x12\x18\n\x14\x41IX_POOLING_UPSAMPLE\x10\x03\x12\x1c\n\x18\x41IX_POOLING_PIXELSHUFFLE\x10\x04\"\x86\x01\n\x0b\x41IXDataType\x12\x12\n\x0e\x41IX_DATA_FLOAT\x10\x00\x12\x13\n\x0f\x41IX_DATA_DOUBLE\x10\x01\x12\x11\n\rAIX_DATA_HALF\x10\x02\x12\x12\n\x0e\x41IX_DATA_UINT8\x10\x03\x12\x12\n\x0e\x41IX_DATA_SINT8\x10\x04\x12\x13\n\x0f\x41IX_DATA_SINT16\x10\x05\"g\n\x0f\x41IXTensorFormat\x12\x13\n\x0f\x41IX_FORMAT_NCHW\x10\x00\x12\x13\n\x0f\x41IX_FORMAT_NHWC\x10\x01\x12\x13\n\x0f\x41IX_FORMAT_NWHC\x10\x02\x12\x15\n\x11\x41IX_FORMAT_VECTOR\x10\x03\"V\n\x08\x41IXGraph\x12\x1d\n\x05layer\x18\x01 \x03(\x0b\x32\x0e.aixh.AIXLayer\x12\x14\n\x0cinput_layers\x18\x02 \x03(\x05\x12\x15\n\routput_layers\x18\x03 \x03(\x05')
)



_AIXLAYER_AIXLAYERTYPE = _descriptor.EnumDescriptor(
  name='AIXLayerType',
  full_name='aixh.AIXLayer.AIXLayerType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='AIX_LAYER_CONVOLUTION', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_LAYER_CONNECTED', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_LAYER_MAXPOOL', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_LAYER_AVGPOOL', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_LAYER_SOFTMAX', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_LAYER_ROUTE', index=5, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_LAYER_REORG', index=6, number=7,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_LAYER_EWADD', index=7, number=8,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_LAYER_UPSAMPLE', index=8, number=9,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_LAYER_PIXELSHUFFLE', index=9, number=10,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_LAYER_GROUP_CONV', index=10, number=11,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_LAYER_SKIP_CONV', index=11, number=12,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_LAYER_ACTIVATION', index=12, number=13,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_LAYER_BATCHNORM', index=13, number=14,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_LAYER_BIASADD', index=14, number=15,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_LAYER_OUTPUT', index=15, number=16,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_LAYER_INPUT', index=16, number=17,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_LAYER_WILDCARD', index=17, number=18,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1182,
  serialized_end=1624,
)
_sym_db.RegisterEnumDescriptor(_AIXLAYER_AIXLAYERTYPE)

_AIXLAYER_AIXACTIVATIONMODE = _descriptor.EnumDescriptor(
  name='AIXActivationMode',
  full_name='aixh.AIXLayer.AIXActivationMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='AIX_ACTIVATION_SIGMOID', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_ACTIVATION_RELU', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_ACTIVATION_LEAKY_RELU', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_ACTIVATION_PRELU', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_ACTIVATION_TANH', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_ACTIVATION_IDENTITY', index=5, number=5,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1627,
  serialized_end=1810,
)
_sym_db.RegisterEnumDescriptor(_AIXLAYER_AIXACTIVATIONMODE)

_AIXLAYER_AIXSAMPLINGMODE = _descriptor.EnumDescriptor(
  name='AIXSamplingMode',
  full_name='aixh.AIXLayer.AIXSamplingMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='AIX_POOLING_MAX', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_POOLING_AVERAGE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_POOLING_REORG', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_POOLING_UPSAMPLE', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_POOLING_PIXELSHUFFLE', index=4, number=4,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1813,
  serialized_end=1955,
)
_sym_db.RegisterEnumDescriptor(_AIXLAYER_AIXSAMPLINGMODE)

_AIXLAYER_AIXDATATYPE = _descriptor.EnumDescriptor(
  name='AIXDataType',
  full_name='aixh.AIXLayer.AIXDataType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='AIX_DATA_FLOAT', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_DATA_DOUBLE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_DATA_HALF', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_DATA_UINT8', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_DATA_SINT8', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_DATA_SINT16', index=5, number=5,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1958,
  serialized_end=2092,
)
_sym_db.RegisterEnumDescriptor(_AIXLAYER_AIXDATATYPE)

_AIXLAYER_AIXTENSORFORMAT = _descriptor.EnumDescriptor(
  name='AIXTensorFormat',
  full_name='aixh.AIXLayer.AIXTensorFormat',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='AIX_FORMAT_NCHW', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_FORMAT_NHWC', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_FORMAT_NWHC', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AIX_FORMAT_VECTOR', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=2094,
  serialized_end=2197,
)
_sym_db.RegisterEnumDescriptor(_AIXLAYER_AIXTENSORFORMAT)


_AIXLAYER_AIXCONVOLUTIONDESC = _descriptor.Descriptor(
  name='AIXConvolutionDesc',
  full_name='aixh.AIXLayer.AIXConvolutionDesc',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dtype', full_name='aixh.AIXLayer.AIXConvolutionDesc.dtype', index=0,
      number=1, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='padding', full_name='aixh.AIXLayer.AIXConvolutionDesc.padding', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stride', full_name='aixh.AIXLayer.AIXConvolutionDesc.stride', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dilation', full_name='aixh.AIXLayer.AIXConvolutionDesc.dilation', index=3,
      number=4, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='groups', full_name='aixh.AIXLayer.AIXConvolutionDesc.groups', index=4,
      number=5, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=726,
  serialized_end=856,
)

_AIXLAYER_AIXSAMPLINGDESC = _descriptor.Descriptor(
  name='AIXSamplingDesc',
  full_name='aixh.AIXLayer.AIXSamplingDesc',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='mode', full_name='aixh.AIXLayer.AIXSamplingDesc.mode', index=0,
      number=1, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='padding', full_name='aixh.AIXLayer.AIXSamplingDesc.padding', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stride', full_name='aixh.AIXLayer.AIXSamplingDesc.stride', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='window', full_name='aixh.AIXLayer.AIXSamplingDesc.window', index=3,
      number=4, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=858,
  serialized_end=970,
)

_AIXLAYER_AIXEWADDDESC = _descriptor.Descriptor(
  name='AIXEWAddDesc',
  full_name='aixh.AIXLayer.AIXEWAddDesc',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='scale', full_name='aixh.AIXLayer.AIXEWAddDesc.scale', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=972,
  serialized_end=1001,
)

_AIXLAYER_AIXTENSOR = _descriptor.Descriptor(
  name='AIXTensor',
  full_name='aixh.AIXLayer.AIXTensor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dtype', full_name='aixh.AIXLayer.AIXTensor.dtype', index=0,
      number=1, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='format', full_name='aixh.AIXLayer.AIXTensor.format', index=1,
      number=2, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dims', full_name='aixh.AIXLayer.AIXTensor.dims', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='fval', full_name='aixh.AIXLayer.AIXTensor.fval', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bval', full_name='aixh.AIXLayer.AIXTensor.bval', index=4,
      number=5, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='size', full_name='aixh.AIXLayer.AIXTensor.size', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ptr', full_name='aixh.AIXLayer.AIXTensor.ptr', index=6,
      number=7, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1004,
  serialized_end=1179,
)

_AIXLAYER = _descriptor.Descriptor(
  name='AIXLayer',
  full_name='aixh.AIXLayer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='aixh.AIXLayer.id', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='aixh.AIXLayer.name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='aixh.AIXLayer.type', index=2,
      number=3, type=14, cpp_type=8, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='preds', full_name='aixh.AIXLayer.preds', index=3,
      number=4, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='succs', full_name='aixh.AIXLayer.succs', index=4,
      number=5, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='input', full_name='aixh.AIXLayer.input', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output', full_name='aixh.AIXLayer.output', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='filter', full_name='aixh.AIXLayer.filter', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bias', full_name='aixh.AIXLayer.bias', index=8,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scale', full_name='aixh.AIXLayer.scale', index=9,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mean', full_name='aixh.AIXLayer.mean', index=10,
      number=11, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='variance', full_name='aixh.AIXLayer.variance', index=11,
      number=12, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='epsilon', full_name='aixh.AIXLayer.epsilon', index=12,
      number=16, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='input_threshold', full_name='aixh.AIXLayer.input_threshold', index=13,
      number=13, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output_threshold', full_name='aixh.AIXLayer.output_threshold', index=14,
      number=14, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='filter_threshold', full_name='aixh.AIXLayer.filter_threshold', index=15,
      number=15, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='convdesc', full_name='aixh.AIXLayer.convdesc', index=16,
      number=20, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='samplingdesc', full_name='aixh.AIXLayer.samplingdesc', index=17,
      number=21, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ewadddesc', full_name='aixh.AIXLayer.ewadddesc', index=18,
      number=22, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='activation', full_name='aixh.AIXLayer.activation', index=19,
      number=30, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_AIXLAYER_AIXCONVOLUTIONDESC, _AIXLAYER_AIXSAMPLINGDESC, _AIXLAYER_AIXEWADDDESC, _AIXLAYER_AIXTENSOR, ],
  enum_types=[
    _AIXLAYER_AIXLAYERTYPE,
    _AIXLAYER_AIXACTIVATIONMODE,
    _AIXLAYER_AIXSAMPLINGMODE,
    _AIXLAYER_AIXDATATYPE,
    _AIXLAYER_AIXTENSORFORMAT,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=21,
  serialized_end=2197,
)


_AIXGRAPH = _descriptor.Descriptor(
  name='AIXGraph',
  full_name='aixh.AIXGraph',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='layer', full_name='aixh.AIXGraph.layer', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='input_layers', full_name='aixh.AIXGraph.input_layers', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output_layers', full_name='aixh.AIXGraph.output_layers', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=2199,
  serialized_end=2285,
)

_AIXLAYER_AIXCONVOLUTIONDESC.fields_by_name['dtype'].enum_type = _AIXLAYER_AIXDATATYPE
_AIXLAYER_AIXCONVOLUTIONDESC.containing_type = _AIXLAYER
_AIXLAYER_AIXSAMPLINGDESC.fields_by_name['mode'].enum_type = _AIXLAYER_AIXSAMPLINGMODE
_AIXLAYER_AIXSAMPLINGDESC.containing_type = _AIXLAYER
_AIXLAYER_AIXEWADDDESC.containing_type = _AIXLAYER
_AIXLAYER_AIXTENSOR.fields_by_name['dtype'].enum_type = _AIXLAYER_AIXDATATYPE
_AIXLAYER_AIXTENSOR.fields_by_name['format'].enum_type = _AIXLAYER_AIXTENSORFORMAT
_AIXLAYER_AIXTENSOR.containing_type = _AIXLAYER
_AIXLAYER.fields_by_name['type'].enum_type = _AIXLAYER_AIXLAYERTYPE
_AIXLAYER.fields_by_name['input'].message_type = _AIXLAYER_AIXTENSOR
_AIXLAYER.fields_by_name['output'].message_type = _AIXLAYER_AIXTENSOR
_AIXLAYER.fields_by_name['filter'].message_type = _AIXLAYER_AIXTENSOR
_AIXLAYER.fields_by_name['bias'].message_type = _AIXLAYER_AIXTENSOR
_AIXLAYER.fields_by_name['scale'].message_type = _AIXLAYER_AIXTENSOR
_AIXLAYER.fields_by_name['mean'].message_type = _AIXLAYER_AIXTENSOR
_AIXLAYER.fields_by_name['variance'].message_type = _AIXLAYER_AIXTENSOR
_AIXLAYER.fields_by_name['convdesc'].message_type = _AIXLAYER_AIXCONVOLUTIONDESC
_AIXLAYER.fields_by_name['samplingdesc'].message_type = _AIXLAYER_AIXSAMPLINGDESC
_AIXLAYER.fields_by_name['ewadddesc'].message_type = _AIXLAYER_AIXEWADDDESC
_AIXLAYER.fields_by_name['activation'].enum_type = _AIXLAYER_AIXACTIVATIONMODE
_AIXLAYER_AIXLAYERTYPE.containing_type = _AIXLAYER
_AIXLAYER_AIXACTIVATIONMODE.containing_type = _AIXLAYER
_AIXLAYER_AIXSAMPLINGMODE.containing_type = _AIXLAYER
_AIXLAYER_AIXDATATYPE.containing_type = _AIXLAYER
_AIXLAYER_AIXTENSORFORMAT.containing_type = _AIXLAYER
_AIXGRAPH.fields_by_name['layer'].message_type = _AIXLAYER
DESCRIPTOR.message_types_by_name['AIXLayer'] = _AIXLAYER
DESCRIPTOR.message_types_by_name['AIXGraph'] = _AIXGRAPH
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

AIXLayer = _reflection.GeneratedProtocolMessageType('AIXLayer', (_message.Message,), {

  'AIXConvolutionDesc' : _reflection.GeneratedProtocolMessageType('AIXConvolutionDesc', (_message.Message,), {
    'DESCRIPTOR' : _AIXLAYER_AIXCONVOLUTIONDESC,
    '__module__' : 'aixh_pb2'
    # @@protoc_insertion_point(class_scope:aixh.AIXLayer.AIXConvolutionDesc)
    })
  ,

  'AIXSamplingDesc' : _reflection.GeneratedProtocolMessageType('AIXSamplingDesc', (_message.Message,), {
    'DESCRIPTOR' : _AIXLAYER_AIXSAMPLINGDESC,
    '__module__' : 'aixh_pb2'
    # @@protoc_insertion_point(class_scope:aixh.AIXLayer.AIXSamplingDesc)
    })
  ,

  'AIXEWAddDesc' : _reflection.GeneratedProtocolMessageType('AIXEWAddDesc', (_message.Message,), {
    'DESCRIPTOR' : _AIXLAYER_AIXEWADDDESC,
    '__module__' : 'aixh_pb2'
    # @@protoc_insertion_point(class_scope:aixh.AIXLayer.AIXEWAddDesc)
    })
  ,

  'AIXTensor' : _reflection.GeneratedProtocolMessageType('AIXTensor', (_message.Message,), {
    'DESCRIPTOR' : _AIXLAYER_AIXTENSOR,
    '__module__' : 'aixh_pb2'
    # @@protoc_insertion_point(class_scope:aixh.AIXLayer.AIXTensor)
    })
  ,
  'DESCRIPTOR' : _AIXLAYER,
  '__module__' : 'aixh_pb2'
  # @@protoc_insertion_point(class_scope:aixh.AIXLayer)
  })
_sym_db.RegisterMessage(AIXLayer)
_sym_db.RegisterMessage(AIXLayer.AIXConvolutionDesc)
_sym_db.RegisterMessage(AIXLayer.AIXSamplingDesc)
_sym_db.RegisterMessage(AIXLayer.AIXEWAddDesc)
_sym_db.RegisterMessage(AIXLayer.AIXTensor)

AIXGraph = _reflection.GeneratedProtocolMessageType('AIXGraph', (_message.Message,), {
  'DESCRIPTOR' : _AIXGRAPH,
  '__module__' : 'aixh_pb2'
  # @@protoc_insertion_point(class_scope:aixh.AIXGraph)
  })
_sym_db.RegisterMessage(AIXGraph)


_AIXLAYER_AIXTENSOR.fields_by_name['bval']._options = None
# @@protoc_insertion_point(module_scope)
