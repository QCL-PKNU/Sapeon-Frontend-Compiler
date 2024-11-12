import numpy as np
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
import onnxruntime as ort
from array import array
import sys

def binary_test(op_name):
    node = helper.make_node(
        op_name,
        inputs=["x", "y"],
        outputs=["z"],
    )

    node_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 5])
    node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 5])
    node_sum = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3 , 4, 5])

    graph = helper.make_graph( nodes=[node], name=op_name+"_test", inputs=[node_x,node_y], outputs=[node_sum])
    model = helper.make_model(graph, producer_name=op_name+'_test')

    np_x = np.random.randn(1, 3, 4, 5).astype(np.float32)
    np_y = np.random.randn(1, 3, 4, 5).astype(np.float32)

    onnx.save(model, "temp.onnx")

    ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    outputs = ort_sess.run(None, {'x': np_x, 'y':np_y})

    output_file = open('tests/test_data/op_' + op_name + '_test_data', 'wb')
    output_file.write(np_x.tobytes())
    output_file.write(np_y.tobytes())
    output_file.write(np.array(outputs).tobytes())
    output_file.close()

    print( np_x[0,0,0,0], np_y[0,0,0,0], outputs[0][0,0,0,0])

def clip_test(op_name):
    if ( op_name == 'default'):
        node = onnx.helper.make_node(
            'Clip',
            inputs=['x'],
            outputs=['y'],
        )
    else:
        node = onnx.helper.make_node(
            'Clip',
            inputs=['x', 'min_v', 'max_v'],
            outputs=['y'],
        )

    min_val = np.float32(-1)
    max_val = np.float32(1)

    node_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 5])
    node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 5])
    node_min = helper.make_tensor_value_info("min_v", TensorProto.FLOAT, ())
    node_max = helper.make_tensor_value_info("max_v", TensorProto.FLOAT, ())


    if ( op_name == 'default'):
        graph = helper.make_graph( nodes=[node], name=op_name+"_test", inputs=[node_x], outputs=[node_y])    
    else:
        graph = helper.make_graph( nodes=[node], name=op_name+"_test", inputs=[node_x,node_min, node_max], outputs=[node_y])    
    
    model = helper.make_model(graph, producer_name=op_name+'_test')

    onnx.save(model, "temp.onnx")

    np_x = np.random.randn(1, 3, 4, 5).astype(np.float32)
    np_x[0,0,0,0] = -2.5;
    np_x[0,0,0,1] = 2.5;

    ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])

    
    if ( op_name == 'default'):
        outputs = ort_sess.run(None, {'x': np_x})
    else:
        outputs = ort_sess.run(None, {'x': np_x, 'min_v':[min_val],'max_v':[max_val]})    

    output_file = open('tests/test_data/op_Clip_' + op_name + '_test_data', 'wb')
    output_file.write(np_x.tobytes())
    output_file.write(np.array(outputs).tobytes())
    output_file.close()

    print( np_x[0,0,0,0], outputs[0][0,0,0,0])
    print( outputs[0][0][0])

    print( np.clip(np_x, min_val, max_val)[0][0] )

def sum_test(op_name):
    data_0 = np.random.randn(1, 4, 4, 1).astype(np.float32)
    data_1 = np.random.randn(1, 1, 4, 1).astype(np.float32)
    data_2 = np.random.randn(1, 1, 1, 4).astype(np.float32)
    result = np.array([6, 9, 12]).astype(np.float32)

    node_data_0 = helper.make_tensor_value_info("data_0", TensorProto.FLOAT, data_0.shape)    
    node_data_1 = helper.make_tensor_value_info("data_1", TensorProto.FLOAT, data_1.shape)        
    node_data_2 = helper.make_tensor_value_info("data_2", TensorProto.FLOAT, data_2.shape)        

    if ( op_name == 'one'):
        temp = data_0
        print(temp.shape)
        # print(temp)
        node = onnx.helper.make_node(
            'Sum',
            inputs=['data_0'],
            outputs=['result'],
        )
        graph = helper.make_graph( nodes=[node], name=op_name+"_test", inputs=[node_data_0], outputs=[node_data_0])    
    elif ( op_name == 'two'):
        temp = np.add(data_0, data_1)        
        print(temp.shape)
        # print(temp)
        node_result = helper.make_tensor_value_info("result", TensorProto.FLOAT, temp.shape)
        node = onnx.helper.make_node(
            'Sum',
            inputs=['data_0', 'data_1'],
            outputs=['result'],
        )
        graph = helper.make_graph( nodes=[node], name=op_name+"_test", inputs=[node_data_0, node_data_1], outputs=[node_result])    
    else:
        temp = np.add(data_0, data_1)
        temp = np.add(temp, data_2)        
        print(temp.shape)
        # print(temp)
        node_result = helper.make_tensor_value_info("result", TensorProto.FLOAT, temp.shape)
        node = onnx.helper.make_node(
            'Sum',
            inputs=['data_0', 'data_1', 'data_2'],
            outputs=['result'],
        )
        graph = helper.make_graph( nodes=[node], name=op_name+"_test", inputs=[node_data_0, node_data_1,node_data_2], outputs=[node_result])    


    
    model = helper.make_model(graph, producer_name=op_name+'_test')

    onnx.save(model, "temp.onnx")

    ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])

    
    output_file = open('tests/test_data/op_Sum_' + op_name + '_test_data', 'wb')
    if ( op_name == 'one'):
        outputs = ort_sess.run(None, {'data_0': data_0})
        output_file.write(data_0.tobytes())
    elif ( op_name == 'two'):
        outputs = ort_sess.run(None, {'data_0': data_0,'data_1': data_1})
        output_file.write(data_0.tobytes())
        output_file.write(data_1.tobytes())
    else:
        outputs = ort_sess.run(None, {'data_0': data_0,'data_1': data_1,'data_2': data_2})
        output_file.write(data_0.tobytes())
        output_file.write(data_1.tobytes())
        output_file.write(data_2.tobytes())    
        
    output_file.write(np.array(outputs).tobytes())
    output_file.close()

    print( outputs[0])    

def unary_test(op_name, is_abs = False ):
    node = helper.make_node(
        op_name,
        inputs=["x"],
        outputs=["y"],
    )

    np_x = np.random.randn(1, 3, 4, 5).astype(np.float32)
    if (is_abs ):
        np_x = np.absolute(np_x)
    np_y = np.sqrt(np_x)

    node_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, np_x.shape)
    node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, np_y.shape)    

    graph = helper.make_graph( nodes=[node], name=op_name+"_test", inputs=[node_x], outputs=[node_y])
    model = helper.make_model(graph, producer_name=op_name+'_test')    

    onnx.save(model, "temp.onnx")

    ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    outputs = ort_sess.run(None, {'x': np_x})

    output_file = open('tests/test_data/op_' + op_name + '_test_data', 'wb')
    output_file.write(np_x.tobytes())
    output_file.write(np.array(outputs).tobytes())
    output_file.close()

    print( outputs[0])

def mish_test(op_name):
    np_x = np.random.randn(1, 3, 4, 5).astype(np.float32)
    np_y = np_x * np.tanh(np.log1p(np.exp(np_x)))

    output_file = open('tests/test_data/op_' + op_name + '_test_data', 'wb')
    output_file.write(np_x.tobytes())
    output_file.write(np_y.tobytes())
    output_file.close()

    print( np_x[0,0,0,0], np_y[0,0,0,0])

def calculate_normalized_shape(X_shape, axis):  # type: ignore
    X_rank = len(X_shape)
    if axis < 0:
        axis = axis + X_rank
    return X_shape[axis:]

# Layer normalization's reference implementation
def _layer_normalization(X, W, B, axis=-1, epsilon=1e-5):  # type: ignore
    X_shape = X.shape
    X_rank = len(X_shape)
    if axis < 0:
        # If axis = -1 and rank of X is 4,
        # the axis is changed to -1 + 4 = 3,
        # which means the last axis.
        axis = axis + X_rank
    unsqueezed_rank = X_rank - axis
    reduction_shape = X_shape[0:axis] + (1,) * unsqueezed_rank

    # Parameter used to convert N-D tensor layer
    # normalization to equivalent 2-D matirx operations.
    row_number = 1
    col_number = 1
    for i in range(X_rank):
        if i < axis:
            row_number *= X_shape[i]
        else:
            col_number *= X_shape[i]

    # After reshaping input tensor X into a matrix,
    # layer normalization is equivalent to conducting
    # standardization on each column vector (s.t. each
    # column has zero mean and unit variance).
    x_mat = np.reshape(X, (row_number, col_number))
    # This computes mean for every x_mat's column.
    x_mean = np.sum(x_mat, axis=1, keepdims=True) / col_number
    x_diff = x_mat - x_mean
    x_squared_diff = x_diff * x_diff
    # This computes variance for every x_mat's column.
    variance = np.sum(x_squared_diff, axis=1, keepdims=True) / col_number
    variance_eps = variance + epsilon
    std_dev = np.sqrt(variance_eps)
    inv_std_dev = np.reciprocal(std_dev)
    # Standardization step. y_mat is zero-mean and unit-variance.
    y_mat = x_diff * inv_std_dev
    # Apply affine transform on normalization outcome.
    # W is linear coefficient while B is bias.
    Y = np.reshape(y_mat, X_shape) * W + B
    # Matrix-level operations' outputs should be reshaped
    # to compensate the initial tensor-to-matrix reshape.
    X_mean = np.reshape(x_mean, reduction_shape)
    X_inv_std_dev = np.reshape(inv_std_dev, reduction_shape)

    return Y, X_mean, X_inv_std_dev

def layer_norm_test(op_name, axis):
    np_x = np.random.randn(1, 3, 4, 5).astype(np.float32)
    normalized_shape = calculate_normalized_shape(np_x.shape, axis)
    np_w = np.random.randn(*normalized_shape).astype(np.float32)
    np_b = np.random.randn(*normalized_shape).astype(np.float32)

    np_y, mean, inv_std_dev = _layer_normalization(np_x, np_w, np_b, axis)

    node = onnx.helper.make_node(
        'LayerNormalization',
        inputs=['X', 'W', "B"],
        outputs=['Y', 'Mean', 'InvStdDev'],
        axis=axis,
    )

    node_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, np_x.shape)
    node_w = helper.make_tensor_value_info("W", TensorProto.FLOAT, np_w.shape)
    node_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, np_b.shape)
    node_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, np_y.shape)

    print( "ShapeX", np_x.shape)
    print( "ShapeW", np_w.shape, np_w.size)
    print( "ShapeB", np_b.shape)
    print( "ShapeY", np_y.shape)
    node_mean = helper.make_tensor_value_info("Mean", TensorProto.FLOAT, mean.shape)
    node_inv_std_dev = helper.make_tensor_value_info("InvStdDev", TensorProto.FLOAT, inv_std_dev.shape)

    graph = helper.make_graph( nodes=[node], name=op_name+ str(axis) + "_test", inputs=[node_x,node_w, node_b], outputs=[node_y,node_mean, node_inv_std_dev])
    model = helper.make_model(graph, producer_name=op_name+ str(axis) + '_test')

    onnx.save(model, "temp.onnx")

    ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    outputs = ort_sess.run(None, {'X': np_x, 'W': np_w, 'B': np_b})

    output_file = open('tests/test_data/op_' + op_name+ str(axis) + '_test_data', 'wb')
    output_file.write(np_x.tobytes())
    output_file.write(np_w.tobytes())
    output_file.write(np_b.tobytes())
    output_file.write(np.array(outputs[0]).tobytes())
    output_file.close()

    print( outputs[0])

def _instancenorm_test_mode(x, s, bias, epsilon=1e-5):  # type: ignore
    dims_x = len(x.shape)
    axis = tuple(range(2, dims_x))
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    return s * (x - mean) / np.sqrt(var + epsilon) + bias

def instance_norm_test(op_name):
    np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    np_s = np.random.randn(3).astype(np.float32)
    np_bias = np.random.randn(3).astype(np.float32)
    np_y = _instancenorm_test_mode(np_x, np_s, np_bias).astype(np.float32)

    node = onnx.helper.make_node(
        "InstanceNormalization",
        inputs=["x", "s", "bias"],
        outputs=["y"],
    )

    node_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, np_x.shape)
    node_s = helper.make_tensor_value_info("s", TensorProto.FLOAT, np_s.shape)
    node_bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, np_bias.shape)
    node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, np_y.shape)

    print( "ShapeX", np_x.shape)
    print( "ShapeS", np_s.shape, np_s.size)
    print( "ShapeBias", np_bias.shape)
    print( "ShapeY", np_y.shape)

    graph = helper.make_graph( nodes=[node], name=op_name+ "_test", inputs=[node_x,node_s, node_bias], outputs=[node_y])
    model = helper.make_model(graph, producer_name=op_name+ '_test')

    onnx.save(model, "temp.onnx")

    ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    outputs = ort_sess.run(None, {'x': np_x, 's': np_s, 'bias': np_bias})

    output_file = open('tests/test_data/op_' + op_name + '_test_data', 'wb')
    output_file.write(np_x.tobytes())
    output_file.write(np_s.tobytes())
    output_file.write(np_bias.tobytes())
    output_file.write(np.array(outputs[0]).tobytes())
    output_file.close()

    print( outputs[0])


def matmul_test( op_name):
    node = onnx.helper.make_node(
        op_name,
        inputs=["a", "b"],
        outputs=["c"],
    )

    np_a = np.random.randn(1, 1, 4, 5).astype(np.float32)
    np_b = np.random.randn(1, 2, 5, 7).astype(np.float32)
    np_c = np.matmul(np_a, np_b)

    node_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, np_a.shape)
    node_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, np_b.shape)
    node_c = helper.make_tensor_value_info("c", TensorProto.FLOAT, np_c.shape)

    graph = helper.make_graph( nodes=[node], name=op_name+ "_test", inputs=[node_a,node_b], outputs=[node_c])
    model = helper.make_model(graph, producer_name=op_name+ '_test')

    onnx.save(model, "temp.onnx")

    ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    outputs = ort_sess.run(None, {'a': np_a, 'b': np_b})

    output_file = open('tests/test_data/op_' + op_name + '_test_data', 'wb')
    output_file.write(np_a.tobytes())
    output_file.write(np_b.tobytes())
    output_file.write(np.array(outputs[0]).tobytes())
    output_file.close()

    print(np_a)
    print(np_b)
    print(outputs[0])

def gemm_reference_implementation(A: np.ndarray, B: np.ndarray, C: np.ndarray = None, alpha: float = 1., beta: float = 1., transA: int = 0,
                                  transB: int = 0) -> np.ndarray:
    A = A if transA == 0 else A.T
    B = B if transB == 0 else B.T
    C = C if C is not None else np.array(0)

    Y = alpha * np.dot(A, B) + beta * C

    return Y

def gemm_test(op_name):
    node = onnx.helper.make_node(
        op_name,
        inputs=["a", "b", "c"],
        outputs=["y"],
        alpha=0.25,
        beta=0.35,
        transA=1,
        transB=1,
    )
    np_a = np.random.ranf([4, 3]).astype(np.float32)
    np_b = np.random.ranf([5, 4]).astype(np.float32)
    np_c = np.random.ranf([1, 1]).astype(np.float32)

    np_y = gemm_reference_implementation(np_a, np_b, np_c, transA=1, transB=1, alpha=0.25, beta=0.35)

    print( np_c)

    node_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, np_a.shape)
    node_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, np_b.shape)
    node_c = helper.make_tensor_value_info("c", TensorProto.FLOAT, np_c.shape)
    node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, np_y.shape)

    graph = helper.make_graph( nodes=[node], name=op_name+ "_test", inputs=[node_a,node_b,node_c], outputs=[node_y])
    model = helper.make_model(graph, producer_name=op_name+ '_test')

    onnx.save(model, "temp.onnx")

    ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    outputs = ort_sess.run(None, {'a': np_a, 'b': np_b, 'c': np_c})

    output_file = open('tests/test_data/op_' + op_name + '_all_test_data', 'wb')
    output_file.write(np_a.tobytes())
    output_file.write(np_b.tobytes())
    output_file.write(np_c.tobytes())
    output_file.write(np.array(outputs[0]).tobytes())
    output_file.close()

    # node = onnx.helper.make_node(
    #     "Gemm", inputs=["a", "b", "c"], outputs=["y"], alpha=0.5
    # )
    
    # np_a = np.random.ranf([3, 5]).astype(np.float32)
    # np_b = np.random.ranf([5, 4]).astype(np.float32)
    # np_c = np.zeros([1, 4]).astype(np.float32)
    # np_y = gemm_reference_implementation(np_a, np_b, np_c, alpha=0.5)

    # node_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, np_a.shape)
    # node_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, np_b.shape)
    # node_c = helper.make_tensor_value_info("c", TensorProto.FLOAT, np_c.shape)
    # node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, np_y.shape)

    # graph = helper.make_graph( nodes=[node], name=op_name+ "_test", inputs=[node_a,node_b,node_c], outputs=[node_y])
    # model = helper.make_model(graph, producer_name=op_name+ '_test')

    # onnx.save(model, "temp.onnx")

    # ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    # outputs = ort_sess.run(None, {'a': np_a, 'b': np_b, 'c': np_c})

    # output_file = open('tests/test_data/op_' + op_name + '_alpha_test_data', 'wb')
    # output_file.write(np_a.tobytes())
    # output_file.write(np_b.tobytes())
    # output_file.write(np_c.tobytes())
    # output_file.write(np.array(outputs[0]).tobytes())
    # output_file.close()

    # node = onnx.helper.make_node(
    #     "Gemm", inputs=["a", "b", "c"], outputs=["y"], beta=0.5
    # )
    
    # np_a = np.random.ranf([2, 7]).astype(np.float32)
    # np_b = np.random.ranf([7, 4]).astype(np.float32)
    # np_c = np.zeros([1, 4]).astype(np.float32)
    # np_y = gemm_reference_implementation(np_a, np_b, np_c, beta=0.5)

    # node_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, np_a.shape)
    # node_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, np_b.shape)
    # node_c = helper.make_tensor_value_info("c", TensorProto.FLOAT, np_c.shape)
    # node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, np_y.shape)

    # graph = helper.make_graph( nodes=[node], name=op_name+ "_test", inputs=[node_a,node_b,node_c], outputs=[node_y])
    # model = helper.make_model(graph, producer_name=op_name+ '_test')

    # onnx.save(model, "temp.onnx")

    # ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    # outputs = ort_sess.run(None, {'a': np_a, 'b': np_b, 'c': np_c})

    # output_file = open('tests/test_data/op_' + op_name + '_beta_test_data', 'wb')
    # output_file.write(np_a.tobytes())
    # output_file.write(np_b.tobytes())
    # output_file.write(np_c.tobytes())
    # output_file.write(np.array(outputs[0]).tobytes())
    # output_file.close()

    # node = onnx.helper.make_node(
    #     "Gemm", inputs=["a", "b", "c"], outputs=["y"]
    # )
    
    # np_a = np.random.ranf([3, 6]).astype(np.float32)
    # np_b = np.random.ranf([6, 4]).astype(np.float32)
    # np_c = np.random.ranf([3, 4]).astype(np.float32)
    # np_y = gemm_reference_implementation(np_a, np_b, np_c)

    # node_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, np_a.shape)
    # node_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, np_b.shape)
    # node_c = helper.make_tensor_value_info("c", TensorProto.FLOAT, np_c.shape)
    # node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, np_y.shape)

    # graph = helper.make_graph( nodes=[node], name=op_name+ "_test", inputs=[node_a,node_b,node_c], outputs=[node_y])
    # model = helper.make_model(graph, producer_name=op_name+ '_test')

    # onnx.save(model, "temp.onnx")

    # ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    # outputs = ort_sess.run(None, {'a': np_a, 'b': np_b, 'c': np_c})

    # output_file = open('tests/test_data/op_' + op_name + '_default_test_data', 'wb')
    # output_file.write(np_a.tobytes())
    # output_file.write(np_b.tobytes())
    # output_file.write(np_c.tobytes())
    # output_file.write(np.array(outputs[0]).tobytes())
    # output_file.close()

    # node = onnx.helper.make_node(
    #     "Gemm", inputs=["a", "b"], outputs=["y"]
    # )
    
    # np_a = np.random.ranf([2, 10]).astype(np.float32)
    # np_b = np.random.ranf([10, 3]).astype(np.float32)
    # np_y = gemm_reference_implementation(np_a, np_b)

    # node_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, np_a.shape)
    # node_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, np_b.shape)
    # node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, np_y.shape)

    # graph = helper.make_graph( nodes=[node], name=op_name+ "_test", inputs=[node_a,node_b], outputs=[node_y])
    # model = helper.make_model(graph, producer_name=op_name+ '_test')

    # onnx.save(model, "temp.onnx")

    # ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    # outputs = ort_sess.run(None, {'a': np_a, 'b': np_b})

    # output_file = open('tests/test_data/op_' + op_name + '_nobias_test_data', 'wb')
    # output_file.write(np_a.tobytes())
    # output_file.write(np_b.tobytes())
    # output_file.write(np.array(outputs[0]).tobytes())
    # output_file.close()

    # node = onnx.helper.make_node(
    #     "Gemm", inputs=["a", "b", "c"], outputs=["y"], transA=1
    # )
    
    # np_a = np.random.ranf([6, 3]).astype(np.float32)
    # np_b = np.random.ranf([6, 4]).astype(np.float32)
    # np_c = np.zeros([1, 4]).astype(np.float32)
    # np_y = gemm_reference_implementation(np_a, np_b, np_c, transA=1)

    # node_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, np_a.shape)
    # node_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, np_b.shape)
    # node_c = helper.make_tensor_value_info("c", TensorProto.FLOAT, np_c.shape)
    # node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, np_y.shape)

    # print( np_b)

    # graph = helper.make_graph( nodes=[node], name=op_name+ "_test", inputs=[node_a,node_b,node_c], outputs=[node_y])
    # model = helper.make_model(graph, producer_name=op_name+ '_test')

    # onnx.save(model, "temp.onnx")

    # ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    # outputs = ort_sess.run(None, {'a': np_a, 'b': np_b, 'c': np_c})

    # output_file = open('tests/test_data/op_' + op_name + '_transA_test_data', 'wb')
    # output_file.write(np_a.tobytes())
    # output_file.write(np_b.tobytes())
    # output_file.write(np_c.tobytes())
    # output_file.write(np.array(outputs[0]).tobytes())
    # output_file.close()

    # node = onnx.helper.make_node(
    #     op_name,
    #     inputs=["a", "b", "c"],
    #     outputs=["y"],
    #     transA=1,
    #     transB=1,
    # )
    # np_a = np.random.ranf([4, 3]).astype(np.float32)
    # np_b = np.random.ranf([5, 4]).astype(np.float32)
    # np_c = np.random.ranf([3, 5]).astype(np.float32)
    
    # np_y = gemm_reference_implementation(np_a, np_b, np_c, transA=1, transB=1)

    # print( np_b)
    # print( np_c)
    
    # node_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, np_a.shape)
    # node_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, np_b.shape)
    # node_c = helper.make_tensor_value_info("c", TensorProto.FLOAT, np_c.shape)
    # node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, np_y.shape)

    # graph = helper.make_graph( nodes=[node], name=op_name+ "_test", inputs=[node_a,node_b,node_c], outputs=[node_y])
    # model = helper.make_model(graph, producer_name=op_name+ '_test')

    # onnx.save(model, "temp.onnx")

    # ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    # outputs = ort_sess.run(None, {'a': np_a, 'b': np_b, 'c': np_c})

    # output_file = open('tests/test_data/op_' + op_name + '_ab_test_data', 'wb')
    # output_file.write(np_a.tobytes())
    # output_file.write(np_b.tobytes())
    # output_file.write(np_c.tobytes())
    # output_file.write(np.array(outputs[0]).tobytes())
    # output_file.close()

    # node = onnx.helper.make_node(
    #     "Gemm", inputs=["a", "b", "c"], outputs=["y"], transB=1
    # )
    
    # np_a = np.random.ranf([3, 6]).astype(np.float32)
    # np_b = np.random.ranf([4, 6]).astype(np.float32)
    # np_c = np.zeros([1, 4]).astype(np.float32)
    # np_y = gemm_reference_implementation(np_a, np_b, np_c, transB=1)

    # node_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, np_a.shape)
    # node_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, np_b.shape)
    # node_c = helper.make_tensor_value_info("c", TensorProto.FLOAT, np_c.shape)
    # node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, np_y.shape)

    # print( np_b)

    # graph = helper.make_graph( nodes=[node], name=op_name+ "_test", inputs=[node_a,node_b,node_c], outputs=[node_y])
    # model = helper.make_model(graph, producer_name=op_name+ '_test')

    # onnx.save(model, "temp.onnx")

    # ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    # outputs = ort_sess.run(None, {'a': np_a, 'b': np_b, 'c': np_c})

    # output_file = open('tests/test_data/op_' + op_name + '_transB_test_data', 'wb')
    # output_file.write(np_a.tobytes())
    # output_file.write(np_b.tobytes())
    # output_file.write(np_c.tobytes())
    # output_file.write(np.array(outputs[0]).tobytes())
    # output_file.close()

    print( outputs[0])

def ReduceSumTest(op_name):
    axes = np.array([1], dtype=np.int64)
    keepdims = 1

    node = onnx.helper.make_node(
        "ReduceSum", inputs=["data", "axes"], outputs=["reduced"], keepdims=keepdims
    )

    data = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
    )
    reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims)

    print( data.shape)
    print( reduced)

    node_axes = helper.make_tensor_value_info("axes", TensorProto.INT64, axes.shape)
    node_data = helper.make_tensor_value_info("data", TensorProto.FLOAT, data.shape)
    node_reduced = helper.make_tensor_value_info("reduced", TensorProto.FLOAT, reduced.shape)

    graph = helper.make_graph( nodes=[node], name=op_name+ "_test", inputs=[node_data,node_axes], outputs=[node_reduced])
    model = helper.make_model(graph, producer_name=op_name+ '_test')

    onnx.save(model, "temp.onnx")

    ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    outputs = ort_sess.run(None, {'data': data, 'axes': axes})

    output_file = open('tests/test_data/op_' + op_name + '_test_data', 'wb')
    output_file.write(data.tobytes())
    output_file.write(axes.tobytes())
    output_file.write(np.array(outputs[0]).tobytes())
    output_file.close()
    print( outputs[0])

    axes = np.array([], dtype=np.int64)
    keepdims = 1

    node = onnx.helper.make_node(
        "ReduceSum", inputs=["data"], outputs=["reduced"], keepdims=keepdims, noop_with_empty_axes=True
    )

    data = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
    )
    reduced = np.array(data)

    print( data.shape)
    print( reduced)

    node_axes = helper.make_tensor_value_info("axes", TensorProto.INT64, axes.shape)
    node_reduced = helper.make_tensor_value_info("reduced", TensorProto.FLOAT, reduced.shape)

    graph = helper.make_graph( nodes=[node], name=op_name+ "_copy_test", inputs=[node_data], outputs=[node_reduced])
    model = helper.make_model(graph, producer_name=op_name+ '_copy_test')

    onnx.save(model, "temp.onnx")

    ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    outputs = ort_sess.run(None, {'data': data})

    output_file = open('tests/test_data/op_' + op_name + '_copy_test_data', 'wb')
    output_file.write(data.tobytes())
    output_file.write(np.array(outputs[0]).tobytes())
    output_file.close()
    print( outputs[0])

def ResizeTest(op_name):
    node = onnx.helper.make_node(
        "Resize",
        inputs=["X", "", "scales"],
        outputs=["Y"],
        mode="linear",
    )

    data = np.array(
        [
            [
                [
                    [1, 2],
                    [3, 4],
                ]
            ]
        ],
        dtype=np.float32,
    )

    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    roi = np.array(
        [
            [
                [
                    [],
                    [],
                    [],
                    [],
                ]
            ]
        ],dtype=np.float32
    )

    output_model = np.array(
        [
            [
                [
                    [1.,   1.25, 1.75, 2.  ],
                    [1.5,  1.75, 2.25, 2.5 ],
                    [2.5,  2.75, 3.25, 3.5 ],
                    [3.,   3.25, 3.75, 4.  ]
                ]
                ]
            ]
        )
    print(data.shape)
    node_data = helper.make_tensor_value_info("X", TensorProto.FLOAT, data.shape)
    node_scales = helper.make_tensor_value_info("scales", TensorProto.FLOAT, scales.shape)
    node_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, (1,1,4,4))

    graph = helper.make_graph( nodes=[node], name=op_name+ "_test", inputs=[node_data,node_scales], outputs=[node_y])
    model = helper.make_model(graph, producer_name=op_name+ '_test')
    onnx.save(model, "temp.onnx")

    ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    outputs = ort_sess.run(None, {'X': data, '':None, 'scales': scales})

    output_file = open('tests/test_data/op_' + op_name + '_test_data', 'wb')
    output_file.write(data.tobytes())
    output_file.write(scales.tobytes())
    output_file.write(np.array(outputs[0]).tobytes())
    output_file.close()
    print( outputs[0])


def ConvTranspose(op_name):
    x = np.array(
        [[[[3.0, 8.0, 1.0], [9.0, 5.0, 7.0], [3.0, 2.0, 6.0]]]]  # (1, 1, 3, 3)
    ).astype(np.float32)
    W = np.array([[[[7.0, 2.0], [1.0, 9.0]]]]).astype(np.float32)  # (1, 1, 2, 2)

    node = onnx.helper.make_node(
        "ConvTranspose", ["X", "W"], ["Y"], dilations=[2, 2]
    )

    y = np.array(
        [
            [
                [
                    [21.0, 56.0, 13.0, 16.0, 2.0],  # [1, 1, 5, 5]
                    [63.0, 35.0, 67.0, 10.0, 14.0],
                    [24.0, 22.0, 76.0, 76.0, 21.0],
                    [9.0, 5.0, 88.0, 45.0, 63.0],
                    [3.0, 2.0, 33.0, 18.0, 54.0],
                ]
            ]
        ]
    ).astype(np.float32)

    print(x.shape)
    print(W.shape)
    print(y.shape)

    dilations = np.array([2, 2]).astype(np.int64)

    print(dilations.shape)

    node_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
    node_w = helper.make_tensor_value_info("W", TensorProto.FLOAT, W.shape)
    node_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, y.shape)

    graph = helper.make_graph( nodes=[node], name=op_name+ "_test", inputs=[node_x,node_w], outputs=[node_y])
    model = helper.make_model(graph, producer_name=op_name+ '_test')
    onnx.save(model, "temp.onnx")

    ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    outputs = ort_sess.run(None, {'X': x, '':None, 'W': W})

    output_file = open('tests/test_data/op_' + op_name + '_test_data', 'wb')
    output_file.write(x.tobytes())
    output_file.write(W.tobytes())
    output_file.write(np.array(outputs[0]).tobytes())
    output_file.close()
    print( outputs[0])

    x = np.array(
        [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]]  # (1, 1, 3, 3)
    ).astype(np.float32)

    W = np.array(
        [
            [
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],  # (1, 2, 3, 3)
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ]
        ]
    ).astype(np.float32)

    node = onnx.helper.make_node(
        "ConvTranspose", ["X", "W"], ["Y"], strides=[3, 2], pads=[1, 2, 1, 2]
    )

    y = np.array(
        [
            [
                [
                    [1.0, 1.0, 3.0],  # (1, 2, 7, 3)
                    [1.0, 1.0, 3.0],
                    [7.0, 4.0, 9.0],
                    [7.0, 4.0, 9.0],
                    [7.0, 4.0, 9.0],
                    [13.0, 7.0, 15.0],
                    [13.0, 7.0, 15.0],
                ],
                [
                    [1.0, 1.0, 3.0],
                    [1.0, 1.0, 3.0],
                    [7.0, 4.0, 9.0],
                    [7.0, 4.0, 9.0],
                    [7.0, 4.0, 9.0],
                    [13.0, 7.0, 15.0],
                    [13.0, 7.0, 15.0],
                ],
            ]
        ]
    ).astype(np.float32)

    print(x.shape)
    print(W.shape)
    print(y.shape)
    
    node_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
    node_w = helper.make_tensor_value_info("W", TensorProto.FLOAT, W.shape)
    node_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, y.shape)

    graph = helper.make_graph( nodes=[node], name=op_name+ "_test", inputs=[node_x,node_w], outputs=[node_y])
    model = helper.make_model(graph, producer_name=op_name+ '_test')
    onnx.save(model, "temp.onnx")

    ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    outputs = ort_sess.run(None, {'X': x, '':None, 'W': W})

    output_file = open('tests/test_data/op_' + op_name + '_pads_test_data', 'wb')
    output_file.write(x.tobytes())
    output_file.write(W.tobytes())
    output_file.write(np.array(outputs[0]).tobytes())
    output_file.close()
    print( outputs[0])

def argmax_use_numpy(data: np.ndarray, axis: int = 0, keepdims: int = 1) -> (np.ndarray):
    result = np.argmax(data, axis=axis)
    if (keepdims == 1):
        result = np.expand_dims(result, axis)
    return result.astype(np.int64)

def arg_test(op_name):
    keepdims = 1
    node = onnx.helper.make_node(
        op_name, inputs=["data"], outputs=["result"], keepdims=keepdims
    )
    
    data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
    # result's shape: [1, 3, 4]
    result = argmax_use_numpy(data, keepdims=keepdims)

    print(data.shape)
    print(result.shape)
    
    node_data = helper.make_tensor_value_info("data", TensorProto.FLOAT, data.shape)
    node_result = helper.make_tensor_value_info("result", TensorProto.INT64, result.shape)    

    graph = helper.make_graph( nodes=[node], name=op_name+ "_test", inputs=[node_data], outputs=[node_result])
    model = helper.make_model(graph, producer_name=op_name+ '_test')
    onnx.save(model, "temp.onnx")

    ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    outputs = ort_sess.run(None, {'data': data })

    output_file = open('tests/test_data/op_' + op_name + '_test_data', 'wb')
    output_file.write(data.tobytes())
    output_file.write(np.array(outputs[0]).tobytes())
    output_file.close()
    print( outputs[0])

    axis = 1
    keepdims = 1
    node = onnx.helper.make_node(
        op_name, inputs=["data"], outputs=["result"], axis=axis, keepdims=keepdims
    )
    result = argmax_use_numpy(data, axis=axis, keepdims=keepdims)
    data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
    # result's shape: [2, 1, 4]
    result = argmax_use_numpy(data, axis=axis, keepdims=keepdims)

    print(data.shape)
    print(result.shape)
    
    node_data = helper.make_tensor_value_info("data", TensorProto.FLOAT, data.shape)
    node_result = helper.make_tensor_value_info("result", TensorProto.INT64, result.shape)    

    graph = helper.make_graph( nodes=[node], name=op_name+ "_test", inputs=[node_data], outputs=[node_result])
    model = helper.make_model(graph, producer_name=op_name+ '_test')
    onnx.save(model, "temp.onnx")

    ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    outputs = ort_sess.run(None, {'data': data })

    output_file = open('tests/test_data/op_' + op_name + '_axis_test_data', 'wb')
    output_file.write(data.tobytes())
    output_file.write(np.array(outputs[0]).tobytes())
    output_file.close()
    print( outputs[0])

def mean_test(op_name):
    data_0 = np.random.randn(4, 4, 1).astype(np.float32)
    data_1 = np.random.randn(1, 4, 1).astype(np.float32)
    data_2 = np.random.randn(1, 1, 4).astype(np.float32)
    result = np.array([6, 9, 12]).astype(np.float32)

    temp = np.add(data_0, data_1)
    temp = np.add(temp, data_2)        
    result = temp / 3.0

    print(result)
    print("----------------")
    
    node = onnx.helper.make_node(
        op_name,
        inputs=["data_0", "data_1", "data_2"],
        outputs=["result"],
    )

    node_data_0 = helper.make_tensor_value_info("data_0", TensorProto.FLOAT, data_0.shape)
    node_data_1 = helper.make_tensor_value_info("data_1", TensorProto.FLOAT, data_1.shape)
    node_data_2 = helper.make_tensor_value_info("data_2", TensorProto.FLOAT, data_2.shape)
    node_result = helper.make_tensor_value_info("result", TensorProto.FLOAT, result.shape)    

    graph = helper.make_graph( nodes=[node], name=op_name+ "_test", inputs=[node_data_0, node_data_1, node_data_2], outputs=[node_result])
    model = helper.make_model(graph, producer_name=op_name+ '_test')
    onnx.save(model, "temp.onnx")

    ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    outputs = ort_sess.run(None, {'data_0': data_0, 'data_1': data_1, 'data_2': data_2 })

    output_file = open('tests/test_data/op_' + op_name + '_test_data', 'wb')
    output_file.write(data_0.tobytes())
    output_file.write(data_1.tobytes())
    output_file.write(data_2.tobytes())
    output_file.write(np.array(outputs[0]).tobytes())
    output_file.close()
    print( outputs[0])

def _batchnorm_test_mode(x, scales, bias, mean, var, epsilon=1e-5):  # type: ignore
    dims_x = len(x.shape)
    dim_ones = (1,) * (dims_x - 2)
    scales = scales.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    mean = mean.reshape(-1, *dim_ones)
    var = var.reshape(-1, *dim_ones)
    return scales * (x - mean) / np.sqrt(var + epsilon) + bias

def batch_norm(op_name):
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    # scales = np.random.randn(3).astype(np.float32)
    scales = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    bias = np.array([0, 0, 0], dtype=np.float32)
    mean = np.random.randn(3).astype(np.float32)
    var = np.random.rand(3).astype(np.float32)
    y = _batchnorm_test_mode(x, scales, bias, mean, var).astype(np.float32)

    node = onnx.helper.make_node(
        "BatchNormalization",
        inputs=["x", "s", "bias", "mean", "var"],
        outputs=["y"],
    )

    node_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape)
    node_scales = helper.make_tensor_value_info("s", TensorProto.FLOAT, scales.shape)
    node_bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, bias.shape)
    node_mean = helper.make_tensor_value_info("mean", TensorProto.FLOAT, mean.shape)    
    node_var = helper.make_tensor_value_info("var", TensorProto.FLOAT, var.shape)    
    node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, y.shape)

    graph = helper.make_graph( nodes=[node], name=op_name+ "_test", inputs=[node_x, node_scales, node_bias, node_mean,node_var ], outputs=[node_y])
    model = helper.make_model(graph, producer_name=op_name+ '_test')
    onnx.save(model, "temp.onnx")

    ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    outputs = ort_sess.run(None, {'x': x, 's': scales, 'bias': bias, 'mean': mean, 'var': var })

    output_file = open('tests/test_data/op_' + op_name + '_test_data', 'wb')
    output_file.write(x.tobytes())
    output_file.write(scales.tobytes())
    output_file.write(bias.tobytes())
    output_file.write(mean.tobytes())
    output_file.write(var.tobytes())
    output_file.write(np.array(outputs[0]).tobytes())
    output_file.close()
    print( outputs[0])

def bias_add(op_name):
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    scales = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    bias = np.random.randn(3).astype(np.float32)
    mean = np.array([0, 0, 0], dtype=np.float32)
    var = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    y = _batchnorm_test_mode(x, scales, bias, mean, var, 0).astype(np.float32)

    node = onnx.helper.make_node(
        "BatchNormalization",
        inputs=["x", "s", "bias", "mean", "var"],
        outputs=["y"],
        epsilon=0.0
    )

    node_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape)
    node_scales = helper.make_tensor_value_info("s", TensorProto.FLOAT, scales.shape)
    node_bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, bias.shape)
    node_mean = helper.make_tensor_value_info("mean", TensorProto.FLOAT, mean.shape)    
    node_var = helper.make_tensor_value_info("var", TensorProto.FLOAT, var.shape)    
    node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, y.shape)

    graph = helper.make_graph( nodes=[node], name=op_name+ "_test", inputs=[node_x, node_scales, node_bias, node_mean,node_var ], outputs=[node_y])
    model = helper.make_model(graph, producer_name=op_name+ '_test')
    onnx.save(model, "temp.onnx")

    ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    outputs = ort_sess.run(None, {'x': x, 's': scales, 'bias': bias, 'mean': mean, 'var': var })

    output_file = open('tests/test_data/op_' + op_name + '_test_data', 'wb')
    output_file.write(x.tobytes())
    output_file.write(scales.tobytes())
    output_file.write(bias.tobytes())
    output_file.write(mean.tobytes())
    output_file.write(var.tobytes())
    output_file.write(np.array(outputs[0]).tobytes())
    output_file.close()
    print( outputs[0])


def conv_test(op_name):
    x = np.array(
        [
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 3, 5, 5) input tensor
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                ],
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                ],
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                ]
            ]
        ]
    ).astype(np.float32)
    W = np.array(
        [
            [
                [
                    [1.0, 2.0, 1.0],  # (3, 3, 3, 3) tensor for convolution weights
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],                                
                [
                    [1.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],                                
                [
                    [1.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],                                
            ],
            [
                [
                    [1.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],                                
                [
                    [1.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],                                
                [
                    [1.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],                                
            ],
            [
                [
                    [1.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],                                
                [
                    [1.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],                                
                [
                    [1.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],                                
            ],
        ]
    ).astype(np.float32)
    
    B = np.array(
        [2.0, 3.0, 4.0]
    ).astype(np.float32)
  
    y = np.array(
        [
            [
                [167.0, 197.0, 227.0, 167.0],
                [317.0, 347.0, 377.0, 272.0],
                [467.0, 497.0, 527.0, 377.0],
                [383.0, 404.0, 425.0, 311.0],
            ],
            [
                [168.0, 198.0, 228.0, 168.0],
                [318.0, 348.0, 378.0, 273.0],
                [468.0, 498.0, 528.0, 378.0],
                [384.0, 405.0, 426.0, 312.0],
            ],
            [
                [169.0, 199.0, 229.0, 169.0],
                [319.0, 349.0, 379.0, 274.0],
                [469.0, 499.0, 529.0, 379.0],
                [385.0, 406.0, 427.0, 313.0],
            ]
        ]
    ).astype(np.float32)

    # Convolution with padding
    node = helper.make_node(
        "Conv",
        inputs=["x", "W", "B"],
        outputs=["y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        dilations=[1, 1],
        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        group=1,
        pads=[0, 0, 1, 1], # pad order : top left bottom right
    )

    print(x.shape)
    print(W.shape)
    print(B.shape)

    node_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape)
    node_W = helper.make_tensor_value_info("W", TensorProto.FLOAT, W.shape)
    node_B = helper.make_tensor_value_info("B", TensorProto.FLOAT, B.shape)
    node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1,3,4,4])

    graph = helper.make_graph( nodes=[node], name=op_name+ "_test", inputs=[node_x, node_W, node_B], outputs=[node_y])
    model = helper.make_model(graph, producer_name=op_name+ '_test')
    onnx.save(model, "temp.onnx")
        
    ort_sess = ort.InferenceSession('temp.onnx', providers=['CPUExecutionProvider'])
    outputs = ort_sess.run(None, {'x': x, 'W': W, 'B': B })

    print(outputs[0].shape)

    output_file = open('tests/test_data/op_' + op_name + '_test_data', 'wb')
    output_file.write(x.tobytes())
    output_file.write(W.tobytes())
    output_file.write(B.tobytes())
    output_file.write(np.array(outputs[0]).tobytes())
    output_file.close()
    print( outputs[0])
    print( outputs[0].shape)


def group_conv_test(op_name):
    x = np.random.randn(3, 9, 10, 10).astype(np.float32)
    W = np.random.randn(60, 3, 3, 3).astype(np.float32)

    y = np.random.randn(3, 60, 10, 10).astype(np.float32)

    # Convolution with padding
    node = onnx.helper.make_node(
        "Conv",
        inputs=["x", "W"],
        outputs=["y"],
        # kernel_shape=[3, 3],
        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        group=3,
        pads=[1, 1, 1, 1],
    )
    
    print(x.shape)
    print(W.shape)

    node_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape)
    node_W = helper.make_tensor_value_info("W", TensorProto.FLOAT, W.shape)
    node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, y.shape)

    graph = helper.make_graph( nodes=[node], name=op_name+ "_test", inputs=[node_x, node_W], outputs=[node_y])
    model = helper.make_model(graph, producer_name=op_name+ '_test')
    onnx.save(model, "temp.onnx")

    ort_sess = ort.InferenceSession('temp.onnx', providers=['CUDAExecutionProvider'])
    outputs = ort_sess.run(None, {'x': x, 'W': W })

    print(outputs[0].shape)

    output_file = open('tests/test_data/op_' + op_name + '_test_data', 'wb')
    output_file.write(x.tobytes())
    output_file.write(W.tobytes())
    output_file.write(np.array(outputs[0]).tobytes())
    output_file.close()
    print( outputs[0])
    print( outputs[0].shape)


def maxpool_test(op_name):
    x = np.array(
        [
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 3, 5, 5) input tensor
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                ],
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                ],
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                ]
            ]
        ]
    ).astype(np.float32)
  
    y = np.array(
        [
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],      # (1, 3, 5, 5) output tensor
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                ],
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],      
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                ],
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],      
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                ],
            ]
        ]
    ).astype(np.float32)

    # Maxpool with padding
    node = helper.make_node(
        "MaxPool",
        inputs=["x"],
        outputs=["y"],
        kernel_shape=[2, 2],
        pads=[0, 0, 1, 1],  # pad order : top left bottom right
    )

    print(x.shape)

    node_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape)
    node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1,3,5,5])

    graph = helper.make_graph(nodes=[node], name=op_name+ "_test", inputs=[node_x], outputs=[node_y])
    model = helper.make_model(graph, producer_name=op_name+ '_test')
    onnx.save(model, "temp.onnx")
        
    ort_sess = ort.InferenceSession('temp.onnx', providers=['CPUExecutionProvider'])
    outputs = ort_sess.run(None, {'x': x })

    print(outputs[0].shape)

    output_file = open('tests/test_data/op_' + op_name + '_test_data', 'wb')
    output_file.write(x.tobytes())
    output_file.write(np.array(outputs[0]).tobytes())
    output_file.close()
    print( outputs[0])
    print( outputs[0].shape)
    

def unary_input(op_name):
    np_x = np.random.randn(1, 3, 4, 4).astype(np.float32)

    output_file = open('tests/test_data/op_' + op_name + '_input_data', 'wb')
    output_file.write(np_x.tobytes())
    output_file.close()
    print( np_x[0,0,0,0])

def unary_input_large(op_name):
    np_x = np.random.randn(1, 36, 4, 4).astype(np.float32)

    output_file = open('tests/test_data/op_' + op_name + '_input_data_large', 'wb')
    output_file.write(np_x.tobytes())
    output_file.close()
    print( np_x[0,0,0,0])

def binary_input(op_name):
    np_x = np.random.randn(1, 3, 4, 4).astype(np.float32)
    np_y = np.random.randn(1, 3, 4, 4).astype(np.float32)

    output_file = open('tests/test_data/op_' + op_name + '_input_data', 'wb')
    output_file.write(np_x.tobytes())
    output_file.write(np_y.tobytes())
    output_file.close()

    print( np_x[0,0,0,0], np_y[0,0,0,0])

def connected_input(op_name):
    x = np.random.randn(3, 3, 10, 10).astype(np.float32)
    W = np.random.randn(60, 3, 3, 3).astype(np.float32)
    
    print(x.shape)
    print(W.shape)

    output_file = open('tests/test_data/op_' + op_name + '_test_data', 'wb')
    output_file.write(x.tobytes())
    output_file.write(W.tobytes())
    output_file.close()

# unary_input("LeakyReLU")
# unary_input("ReLU")
# unary_input("Sigmoid")
# unary_input("SkipConvolution")
# unary_input("Softmax")
# unary_input("Input")
# unary_input("Output")
# unary_input("Avgpool")
# unary_input("Maxpool")
# unary_input_large("Pixelshuffle")
# unary_input("Reorg")
# unary_input("Upsample")
# binary_input("EWAdd")
# binary_input("Route")
# connected_input("Connected")

#binary_test("Add")
# binary_test("Mul")
# binary_test("Sub")
# binary_test("PRelu")

# clip_test("default")
# clip_test("full")

# sum_test("one")
# sum_test("two")
# sum_test("three")
# mish_test("Mish")
# unary_test("Celu")
# unary_test("Selu")

# layer_norm_test("LayerNorm", -1)
# layer_norm_test("LayerNorm", 1)

# instance_norm_test("InstanceNorm")

# matmul_test( "MatMul")

#gemm_test("Gemm")

#ReduceSumTest("ReduceSum")

# ResizeTest("Resize")

# ConvTranspose("ConvTranspose")

#arg_test("ArgMax")
#arg_test("ArgMin")

# unary_test("Elu")
# unary_test("Sqrt", True)

# mean_test("Mean")

# batch_norm("BatchNormalization")

# bias_add("BiasAdd")

# conv_test("Conv")

# group_conv_test("GroupConv")

maxpool_test("Maxpool")