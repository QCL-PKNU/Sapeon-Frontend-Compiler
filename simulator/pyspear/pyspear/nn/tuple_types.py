from typing import Union, Tuple

_int_scalar_t = int
_int_2_tuple_t = Tuple[int, int]
_int_4_tuple_t = Tuple[int, int, int, int]

_float_scalar_t = float
_float_2_tuple_t = Tuple[float, float]

_int_scalar_or_tuple_2_t = Union[_int_scalar_t, _int_2_tuple_t]
_int_scalar_or_tuple_2_or_4_t = Union[
    _int_scalar_t,
    _int_2_tuple_t,
    _int_4_tuple_t,
]

_float_scalar_or_tuple_2_t = Union[_float_scalar_t, _float_2_tuple_t]

kernel_size_t = _int_scalar_or_tuple_2_t

padding_t = _int_scalar_or_tuple_2_or_4_t
stride_t = _int_scalar_or_tuple_2_t
window_t = _int_scalar_or_tuple_2_t
dilation_t = _int_scalar_or_tuple_2_t
scale_t = _float_scalar_or_tuple_2_t
