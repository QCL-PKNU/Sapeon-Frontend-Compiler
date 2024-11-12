import numpy as np
from onnx import helper, save, TensorProto
import onnxruntime as ort


def convert_op_name_to_onnx_op_name(op_name: str):
    # EWAdd == Add, EWMul == Mul
    if op_name.startswith("EW"):
        return op_name[2:]
    elif op_name == "Route":
        return "Concat"
    elif op_name == "Avgpool":
        return "GlobalAveragePool"
    elif op_name == "Upsample":
        return "Resize"
    elif op_name == "Reorg":
        return "SpaceToDepth"
    elif op_name == "Pixelshuffle":
        return "DepthToSpace"

    return op_name


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
                ],
            ]
        ]
    ).astype(np.float32)
    W = np.array(
        [
            [
                [
                    [
                        1.0,
                        2.0,
                        1.0,
                    ],  # (3, 3, 3, 3) weights tensor
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

    B = np.array([2.0, 3.0, 4.0]).astype(np.float32)

    # Convolution with padding
    node = helper.make_node(
        "Conv",
        inputs=["x", "W", "B"],
        outputs=["y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        dilations=[1, 1],
        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        pads=[0, 0, 1, 1],  # pad order : top left bottom right
    )

    node_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape)
    node_W = helper.make_tensor_value_info("W", TensorProto.FLOAT, W.shape)
    node_B = helper.make_tensor_value_info("B", TensorProto.FLOAT, B.shape)
    node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 4])

    graph = helper.make_graph(
        nodes=[node],
        name=op_name + "_test",
        inputs=[node_x, node_W, node_B],
        outputs=[node_y],
    )
    model = helper.make_model(graph, producer_name=op_name + "_test")
    save(model, "temp.onnx")

    ort_sess = ort.InferenceSession(
        "temp.onnx", providers=["CPUExecutionProvider"]
    )
    outputs = ort_sess.run(None, {"x": x, "W": W, "B": B})

    output_file = open("tests/test_data/op_" + op_name + "_test_data", "wb")
    output_file.write(x.tobytes())
    output_file.write(W.tobytes())
    output_file.write(B.tobytes())
    output_file.write(np.array(outputs[0]).tobytes())
    output_file.close()


def group_conv_test(op_name):
    x = np.random.randn(3, 9, 10, 10).astype(np.float32)
    W = np.random.randn(60, 3, 3, 3).astype(np.float32)

    y = np.random.randn(3, 60, 10, 10).astype(np.float32)

    # Convolution with padding
    node = helper.make_node(
        "Conv",
        inputs=["x", "W"],
        outputs=["y"],
        # kernel_shape=[3, 3],
        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        group=3,
        pads=[1, 1, 1, 1],
    )

    node_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape)
    node_W = helper.make_tensor_value_info("W", TensorProto.FLOAT, W.shape)
    node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, y.shape)

    graph = helper.make_graph(
        nodes=[node],
        name=op_name + "_test",
        inputs=[node_x, node_W],
        outputs=[node_y],
    )
    model = helper.make_model(graph, producer_name=op_name + "_test")
    save(model, "temp.onnx")

    ort_sess = ort.InferenceSession(
        "temp.onnx", providers=["CPUExecutionProvider"]
    )
    outputs = ort_sess.run(None, {"x": x, "W": W})

    output_file = open("tests/test_data/op_" + op_name + "_test_data", "wb")
    output_file.write(x.tobytes())
    output_file.write(W.tobytes())
    output_file.write(np.array(outputs[0]).tobytes())
    output_file.close()


def connected_test(op_name):
    x = np.random.randn(1, 4096).astype(np.float32)
    W = np.random.randn(1000, 4096).astype(np.float32)
    B = np.random.randn(1000).astype(np.float32)

    y = np.random.randn(1, 1000).astype(np.float32)

    # Convolution with padding
    node = helper.make_node(
        "Gemm",
        inputs=["x", "W"],
        outputs=["y"],
        transB=1,
    )

    node_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape)
    node_W = helper.make_tensor_value_info("W", TensorProto.FLOAT, W.shape)
    node_B = helper.make_tensor_value_info("B", TensorProto.FLOAT, B.shape)
    node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, y.shape)

    graph = helper.make_graph(
        nodes=[node],
        name=op_name + "_test",
        inputs=[node_x, node_W, node_B],
        outputs=[node_y],
    )
    model = helper.make_model(graph, producer_name=op_name + "_test")
    save(model, "temp.onnx")

    ort_sess = ort.InferenceSession(
        "temp.onnx", providers=["CPUExecutionProvider"]
    )
    outputs = ort_sess.run(None, {"x": x, "W": W, "B": B})

    output_file = open("tests/test_data/op_" + op_name + "_test_data", "wb")
    output_file.write(x.tobytes())
    output_file.write(W.tobytes())
    output_file.write(B.tobytes())
    output_file.write(np.array(outputs[0]).tobytes())
    output_file.close()


def binary_test(op_name: str):
    onnx_op_name = convert_op_name_to_onnx_op_name(op_name)

    node = helper.make_node(
        onnx_op_name,
        inputs=["x", "y"],
        outputs=["z"],
    )

    node_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 5])
    node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 5])
    node_output = helper.make_tensor_value_info(
        "z", TensorProto.FLOAT, [1, 3, 5, 5]
    )

    graph = helper.make_graph(
        nodes=[node],
        name=op_name + "_test",
        inputs=[node_x, node_y],
        outputs=[node_output],
    )
    model = helper.make_model(graph, producer_name=op_name + "_test")

    np_x = np.random.randn(1, 3, 5, 5).astype(np.float32)
    np_y = np.random.randn(1, 3, 5, 5).astype(np.float32)

    save(model, "temp.onnx")

    ort_sess = ort.InferenceSession(
        "temp.onnx", providers=["CPUExecutionProvider"]
    )
    outputs = ort_sess.run(None, {"x": np_x, "y": np_y})

    output_file = open("tests/test_data/op_" + op_name + "_test_data", "wb")
    output_file.write(np_x.tobytes())
    output_file.write(np_y.tobytes())
    output_file.write(np.array(outputs).tobytes())
    output_file.close()

    print(np_x[0, 0, 0, 0], np_y[0, 0, 0, 0], outputs[0][0, 0, 0, 0])


def route_test(op_name: str):
    node = helper.make_node(
        "Concat",
        axis=1,
        inputs=["x", "y"],
        outputs=["z"],
    )

    node_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 5])
    node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 5])
    node_output = helper.make_tensor_value_info(
        "z", TensorProto.FLOAT, [1, 6, 5, 5]
    )

    graph = helper.make_graph(
        nodes=[node],
        name=op_name + "_test",
        inputs=[node_x, node_y],
        outputs=[node_output],
    )
    model = helper.make_model(graph, producer_name=op_name + "_test")

    np_x = np.random.randn(1, 3, 5, 5).astype(np.float32)
    np_y = np.random.randn(1, 3, 5, 5).astype(np.float32)

    save(model, "temp.onnx")

    ort_sess = ort.InferenceSession(
        "temp.onnx", providers=["CPUExecutionProvider"]
    )
    outputs = ort_sess.run(None, {"x": np_x, "y": np_y})

    output_file = open("tests/test_data/op_" + op_name + "_test_data", "wb")
    output_file.write(np_x.tobytes())
    output_file.write(np_y.tobytes())
    output_file.write(np.array(outputs).tobytes())
    output_file.close()

    print(np_x[0, 0, 0, 0], np_y[0, 0, 0, 0], outputs[0][0, 0, 0, 0])


def simple_unary_test(op_name):
    onnx_op_name = convert_op_name_to_onnx_op_name(op_name)
    node = helper.make_node(onnx_op_name, inputs=["x"], outputs=["y"])

    node_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 5])
    node_output = helper.make_tensor_value_info(
        "y", TensorProto.FLOAT, [1, 3, 5, 5]
    )

    graph = helper.make_graph(
        nodes=[node],
        name=op_name + "_test",
        inputs=[node_x],
        outputs=[node_output],
    )
    model = helper.make_model(graph, producer_name=op_name + "_test")

    np_x = np.random.randn(1, 3, 5, 5).astype(np.float32)

    save(model, "temp.onnx")

    ort_sess = ort.InferenceSession(
        "temp.onnx", providers=["CPUExecutionProvider"]
    )
    outputs = ort_sess.run(None, {"x": np_x})

    output_file = open("tests/test_data/op_" + op_name + "_test_data", "wb")
    output_file.write(np_x.tobytes())
    output_file.write(np.array(outputs).tobytes())
    output_file.close()


def leaky_relu_test(op_name):
    onnx_op_name = convert_op_name_to_onnx_op_name(op_name)
    node = helper.make_node(
        onnx_op_name,
        inputs=["x"],
        outputs=["y"],
        alpha=0.1,
    )

    node_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 5])
    node_output = helper.make_tensor_value_info(
        "y", TensorProto.FLOAT, [1, 3, 5, 5]
    )

    graph = helper.make_graph(
        nodes=[node],
        name=op_name + "_test",
        inputs=[node_x],
        outputs=[node_output],
    )
    model = helper.make_model(graph, producer_name=op_name + "_test")

    np_x = np.random.randn(1, 3, 5, 5).astype(np.float32)

    save(model, "temp.onnx")

    ort_sess = ort.InferenceSession(
        "temp.onnx", providers=["CPUExecutionProvider"]
    )
    outputs = ort_sess.run(None, {"x": np_x})

    output_file = open("tests/test_data/op_" + op_name + "_test_data", "wb")
    output_file.write(np_x.tobytes())
    output_file.write(np.array(outputs).tobytes())
    output_file.close()


def pixelshuffle_test(op_name):
    onnx_op_name = convert_op_name_to_onnx_op_name(op_name)
    node = helper.make_node(
        onnx_op_name,
        inputs=["x"],
        outputs=["y"],
        blocksize=3,
    )

    node_x = helper.make_tensor_value_info(
        "x", TensorProto.FLOAT, [1, 36, 4, 4]
    )
    node_output = helper.make_tensor_value_info(
        "y", TensorProto.FLOAT, [1, 4, 12, 12]
    )

    graph = helper.make_graph(
        nodes=[node],
        name=op_name + "_test",
        inputs=[node_x],
        outputs=[node_output],
    )
    model = helper.make_model(graph, producer_name=op_name + "_test")

    np_x = np.random.randn(1, 36, 4, 4).astype(np.float32)

    save(model, "temp.onnx")

    ort_sess = ort.InferenceSession(
        "temp.onnx", providers=["CPUExecutionProvider"]
    )
    outputs = ort_sess.run(None, {"x": np_x})

    output_file = open("tests/test_data/op_" + op_name + "_test_data", "wb")
    output_file.write(np_x.tobytes())
    output_file.write(np.array(outputs).tobytes())
    output_file.close()


def relu6_test(op_name):
    node = helper.make_node("Relu", inputs=["x"], outputs=["y"])

    node_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 5])
    node_output = helper.make_tensor_value_info(
        "y", TensorProto.FLOAT, [1, 3, 5, 5]
    )

    graph = helper.make_graph(
        nodes=[node],
        name=op_name + "_test",
        inputs=[node_x],
        outputs=[node_output],
    )
    model = helper.make_model(graph, producer_name=op_name + "_test")

    orig_np_x = np.random.randn(1, 3, 5, 5).astype(np.float32)
    np_x = np.clip(orig_np_x, np.float32(0), np.float32(6))

    save(model, "temp.onnx")

    ort_sess = ort.InferenceSession(
        "temp.onnx", providers=["CPUExecutionProvider"]
    )
    outputs = ort_sess.run(None, {"x": np_x})

    output_file = open("tests/test_data/op_" + op_name + "_test_data", "wb")
    output_file.write(orig_np_x.tobytes())
    output_file.write(np.array(outputs).tobytes())
    output_file.close()


def softmax_test(op_name):
    onnx_op_name = convert_op_name_to_onnx_op_name(op_name)
    node = helper.make_node(onnx_op_name, inputs=["x"], outputs=["y"], axis=1)

    node_x = helper.make_tensor_value_info(
        "x", TensorProto.FLOAT, [1, 1000, 1, 1]
    )
    node_output = helper.make_tensor_value_info(
        "y", TensorProto.FLOAT, [1, 1000, 1, 1]
    )

    graph = helper.make_graph(
        nodes=[node],
        name=op_name + "_test",
        inputs=[node_x],
        outputs=[node_output],
    )
    model = helper.make_model(graph, producer_name=op_name + "_test")

    np_x = np.random.randn(1, 1000, 1, 1).astype(np.float32)

    save(model, "temp.onnx")

    ort_sess = ort.InferenceSession(
        "temp.onnx", providers=["CPUExecutionProvider"]
    )
    outputs = ort_sess.run(None, {"x": np_x})

    output_file = open("tests/test_data/op_" + op_name + "_test_data", "wb")
    output_file.write(np_x.tobytes())
    output_file.write(np.array(outputs).tobytes())
    output_file.close()


def avgpool_test(op_name):
    onnx_op_name = convert_op_name_to_onnx_op_name(op_name)
    node = helper.make_node(onnx_op_name, inputs=["x"], outputs=["y"])

    node_x = helper.make_tensor_value_info(
        "x", TensorProto.FLOAT, [1, 3, 14, 14]
    )
    node_output = helper.make_tensor_value_info(
        "y", TensorProto.FLOAT, [1, 3, 1, 1]
    )

    graph = helper.make_graph(
        nodes=[node],
        name=op_name + "_test",
        inputs=[node_x],
        outputs=[node_output],
    )
    model = helper.make_model(graph, producer_name=op_name + "_test")

    np_x = np.random.randn(1, 3, 14, 14).astype(np.float32)

    save(model, "temp.onnx")

    ort_sess = ort.InferenceSession(
        "temp.onnx", providers=["CPUExecutionProvider"]
    )
    outputs = ort_sess.run(None, {"x": np_x})

    output_file = open("tests/test_data/op_" + op_name + "_test_data", "wb")
    output_file.write(np_x.tobytes())
    output_file.write(np.array(outputs).tobytes())
    output_file.close()


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
                ],
            ]
        ]
    ).astype(np.float32)

    y = np.array(
        [
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 3, 5, 5) output tensor
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

    node_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape)
    node_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 5])

    graph = helper.make_graph(
        nodes=[node], name=op_name + "_test", inputs=[node_x], outputs=[node_y]
    )
    model = helper.make_model(graph, producer_name=op_name + "_test")
    save(model, "temp.onnx")

    ort_sess = ort.InferenceSession(
        "temp.onnx", providers=["CPUExecutionProvider"]
    )
    outputs = ort_sess.run(None, {"x": x})

    output_file = open("tests/test_data/op_" + op_name + "_test_data", "wb")
    output_file.write(x.tobytes())
    output_file.write(np.array(outputs[0]).tobytes())
    output_file.close()


def upsample_test(op_name):
    onnx_op_name = convert_op_name_to_onnx_op_name(op_name)
    node = helper.make_node(
        onnx_op_name,
        inputs=["x", "", "scales"],
        outputs=["y"],
        mode="nearest",
    )

    node_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 5])
    node_scales = helper.make_tensor_value_info(
        "scales", TensorProto.FLOAT, [4]
    )
    node_output = helper.make_tensor_value_info(
        "y", TensorProto.FLOAT, [1, 3, 10, 10]
    )

    graph = helper.make_graph(
        nodes=[node],
        name=op_name + "_test",
        inputs=[node_x, node_scales],
        outputs=[node_output],
    )
    model = helper.make_model(graph, producer_name=op_name + "_test")

    np_x = np.random.randn(1, 3, 5, 5).astype(np.float32)
    np_scales = np.array([1.0, 1.0, 2.0, 2.0]).astype(np.float32)

    save(model, "temp.onnx")

    ort_sess = ort.InferenceSession(
        "temp.onnx", providers=["CPUExecutionProvider"]
    )
    outputs = ort_sess.run(None, {"x": np_x, "scales": np_scales})

    output_file = open("tests/test_data/op_" + op_name + "_test_data", "wb")
    output_file.write(np_x.tobytes())
    output_file.write(np.array(outputs).tobytes())
    output_file.close()


def reorg_test(op_name):
    onnx_op_name = convert_op_name_to_onnx_op_name(op_name)
    node = helper.make_node(
        onnx_op_name,
        inputs=["x"],
        outputs=["y"],
        blocksize=2,
    )

    node_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 6])
    node_output = helper.make_tensor_value_info(
        "y", TensorProto.FLOAT, [1, 4, 2, 3]
    )

    graph = helper.make_graph(
        nodes=[node],
        name=op_name + "_test",
        inputs=[node_x],
        outputs=[node_output],
    )
    model = helper.make_model(graph, producer_name=op_name + "_test")

    np_x = np.random.randn(1, 1, 4, 6).astype(np.float32)

    save(model, "temp.onnx")

    ort_sess = ort.InferenceSession(
        "temp.onnx", providers=["CPUExecutionProvider"]
    )
    outputs = ort_sess.run(None, {"x": np_x})

    output_file = open("tests/test_data/op_" + op_name + "_test_data", "wb")
    output_file.write(np_x.tobytes())
    output_file.write(np.array(outputs).tobytes())
    output_file.close()


if __name__ == "__main__":
    conv_test("Conv")
    group_conv_test("GroupConv")
    connected_test("Connected")
    binary_test("EWAdd")
    binary_test("EWMul")
    route_test("Route")
    leaky_relu_test("LeakyRelu")
    simple_unary_test("Relu")
    relu6_test("Relu6")
    simple_unary_test("Sigmoid")
    pixelshuffle_test("Pixelshuffle")
    avgpool_test("Avgpool")
    maxpool_test("Maxpool")
    upsample_test("Upsample")
    reorg_test("Reorg")

    # softmax implementation is different
    softmax_test("Softmax")

    # now onnx opset 15 version but need onnx opset 18 version
    # simple_unary_test("Mish")
