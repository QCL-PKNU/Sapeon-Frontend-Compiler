from typing import List

from pyspear import nn
from pyspear.graph import GraphSpearV1


def create_nodes() -> List[nn.Node]:
    results = []
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
    # conv1.load_filter_weight(
    #     "/workspaces/aixgraph_simulator/pyspear/weights.txt"
    # )
    # conv1.load_filter_bias("/workspaces/aixgraph_simulator/pyspear/bias.txt")
    results.append(conv0)

    maxpool1 = nn.Maxpool(
        node_id=1,
        name="/maxpool/MaxPool_output_0",
        input_shapes=[(1, 64, 112, 112)],
        output_shape=(1, 64, 56, 56),
        padding=1,
        window=3,
        stride=2,
    )
    results.append(maxpool1)

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
    results.append(conv2)

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
    results.append(conv3)

    ewadd4 = nn.Ewadd(
        node_id=4,
        name="/layer1/layer1.0/relu_1/Relu_output_0",
        input_shapes=[(1, 64, 56, 56), (1, 64, 56, 56)],
        output_shape=(1, 64, 56, 56),
        activation=nn.ReLU(),
    )
    results.append(ewadd4)

    conv5 = nn.Conv(
        node_id=5,
        name="/layer1/layer1.1/relu/Relu_output_0",
        input_shapes=[(1, 64, 56, 56)],
        output_shape=(1, 64, 56, 56),
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        padding=1,
        stride=1,
        activation=nn.ReLU(),
    )
    results.append(conv5)

    conv6 = nn.Conv(
        node_id=6,
        name="/layer1/layer1.0/conv2/Conv_output_0",
        input_shapes=[(1, 64, 56, 56)],
        output_shape=(1, 64, 56, 56),
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        padding=1,
        stride=1,
    )
    results.append(conv6)

    ewadd7 = nn.Ewadd(
        node_id=7,
        name="/layer1/layer1.0/relu_1/Relu_output_0",
        input_shapes=[(1, 64, 56, 56), (1, 64, 56, 56)],
        output_shape=(1, 64, 56, 56),
        activation=nn.ReLU(),
    )
    results.append(ewadd7)

    conv8 = nn.Conv(
        node_id=8,
        name="/layer2/layer2.0/relu/Relu_output_0",
        input_shapes=[(1, 64, 56, 56)],
        output_shape=(1, 128, 28, 28),
        in_channels=64,
        out_channels=128,
        kernel_size=3,
        padding=1,
        stride=2,
        activation=nn.ReLU(),
    )
    results.append(conv8)

    conv9 = nn.Conv(
        node_id=9,
        name="/layer2/layer2.0/downsample/downsample.0/Conv_output_0",
        input_shapes=[(1, 64, 56, 56)],
        output_shape=(1, 128, 28, 28),
        in_channels=64,
        out_channels=128,
        kernel_size=1,
        padding=0,
        stride=2,
    )
    results.append(conv9)

    conv10 = nn.Conv(
        node_id=10,
        name="/layer2/layer2.0/conv2/Conv_output_0",
        input_shapes=[(1, 128, 28, 28)],
        output_shape=(1, 128, 28, 28),
        in_channels=128,
        out_channels=128,
        kernel_size=3,
        padding=1,
        stride=1,
    )
    results.append(conv10)

    ewadd11 = nn.Ewadd(
        node_id=11,
        name="/layer2/layer2.0/relu_1/Relu_output_0",
        input_shapes=[(1, 128, 28, 28), (1, 128, 28, 28)],
        output_shape=(1, 128, 28, 28),
        activation=nn.ReLU(),
    )
    results.append(ewadd11)

    conv12 = nn.Conv(
        node_id=12,
        name="/layer2/layer2.1/relu/Relu_output_0",
        input_shapes=[(1, 128, 28, 28)],
        output_shape=(1, 128, 28, 28),
        in_channels=128,
        out_channels=128,
        kernel_size=3,
        padding=1,
        stride=1,
        activation=nn.ReLU(),
    )
    results.append(conv12)

    conv13 = nn.Conv(
        node_id=13,
        name="/layer2/layer2.1/conv2/Conv_output_0",
        input_shapes=[(1, 128, 28, 28)],
        output_shape=(1, 128, 28, 28),
        in_channels=128,
        out_channels=128,
        kernel_size=3,
        padding=1,
        stride=1,
    )
    results.append(conv13)

    ewadd14 = nn.Ewadd(
        node_id=14,
        name="/layer2/layer2.1/relu_1/Relu_output_0",
        input_shapes=[(1, 128, 28, 28), (1, 128, 28, 28)],
        output_shape=(1, 128, 28, 28),
        activation=nn.ReLU(),
    )
    results.append(ewadd14)

    conv15 = nn.Conv(
        node_id=15,
        name="/layer3/layer3.0/relu/Relu_output_0",
        input_shapes=[(1, 128, 28, 28)],
        output_shape=(1, 256, 14, 14),
        in_channels=128,
        out_channels=256,
        kernel_size=3,
        padding=1,
        stride=2,
        activation=nn.ReLU(),
    )
    results.append(conv15)

    conv16 = nn.Conv(
        node_id=16,
        name="/layer3/layer3.0/downsample/downsample.0/Conv_output_0",
        input_shapes=[(1, 128, 28, 28)],
        output_shape=(1, 256, 14, 14),
        in_channels=128,
        out_channels=256,
        kernel_size=1,
        padding=0,
        stride=2,
    )
    results.append(conv16)

    conv17 = nn.Conv(
        node_id=17,
        name="/layer3/layer3.0/conv2/Conv_output_0",
        input_shapes=[(1, 256, 14, 14)],
        output_shape=(1, 256, 14, 14),
        in_channels=256,
        out_channels=256,
        kernel_size=3,
        padding=1,
        stride=1,
    )
    results.append(conv17)

    ewadd18 = nn.Ewadd(
        node_id=18,
        name="/layer3/layer3.0/relu_1/Relu_output_0",
        input_shapes=[(1, 256, 14, 14), (1, 256, 14, 14)],
        output_shape=(1, 256, 14, 14),
        activation=nn.ReLU(),
    )
    results.append(ewadd18)

    conv19 = nn.Conv(
        node_id=19,
        name="/layer3/layer3.1/relu/Relu_output_0",
        input_shapes=[(1, 256, 14, 14)],
        output_shape=(1, 256, 14, 14),
        in_channels=256,
        out_channels=256,
        kernel_size=3,
        padding=1,
        stride=1,
        activation=nn.ReLU(),
    )
    results.append(conv19)

    conv20 = nn.Conv(
        node_id=20,
        name="/layer3/layer3.1/conv2/Conv_output_0",
        input_shapes=[(1, 256, 14, 14)],
        output_shape=(1, 256, 14, 14),
        in_channels=256,
        out_channels=256,
        kernel_size=3,
        padding=1,
        stride=1,
    )
    results.append(conv20)

    ewadd21 = nn.Ewadd(
        node_id=21,
        name="/layer3/layer3.1/relu_1/Relu_output_0",
        input_shapes=[(1, 256, 14, 14), (1, 256, 14, 14)],
        output_shape=(1, 256, 14, 14),
        activation=nn.ReLU(),
    )
    results.append(ewadd21)

    conv22 = nn.Conv(
        node_id=22,
        name="/layer4/layer4.0/relu/Relu_output_0",
        input_shapes=[(1, 256, 14, 14)],
        output_shape=(1, 512, 7, 7),
        in_channels=256,
        out_channels=512,
        kernel_size=3,
        padding=1,
        stride=2,
        activation=nn.ReLU(),
    )
    results.append(conv22)

    conv23 = nn.Conv(
        node_id=23,
        name="/layer4/layer4.0/downsample/downsample.0/Conv_output_0",
        input_shapes=[(1, 256, 14, 14)],
        output_shape=(1, 512, 7, 7),
        in_channels=256,
        out_channels=512,
        kernel_size=1,
        padding=0,
        stride=2,
    )
    results.append(conv23)

    conv24 = nn.Conv(
        node_id=24,
        name="/layer4/layer4.0/conv2/Conv_output_0",
        input_shapes=[(1, 512, 7, 7)],
        output_shape=(1, 512, 7, 7),
        in_channels=512,
        out_channels=512,
        kernel_size=3,
        padding=1,
        stride=1,
    )
    results.append(conv24)

    ewadd25 = nn.Ewadd(
        node_id=25,
        name="/layer4/layer4.0/relu_1/Relu_output_0",
        input_shapes=[(1, 512, 7, 7), (1, 512, 7, 7)],
        output_shape=(1, 512, 7, 7),
        activation=nn.ReLU(),
    )
    results.append(ewadd25)

    conv26 = nn.Conv(
        node_id=26,
        name="/layer4/layer4.1/relu/Relu_output_0",
        input_shapes=[(1, 512, 7, 7)],
        output_shape=(1, 512, 7, 7),
        in_channels=512,
        out_channels=512,
        kernel_size=3,
        padding=1,
        stride=1,
        activation=nn.ReLU(),
    )
    results.append(conv26)

    conv27 = nn.Conv(
        node_id=27,
        name="/layer4/layer4.1/conv2/Conv_output_0",
        input_shapes=[(1, 512, 7, 7)],
        output_shape=(1, 512, 7, 7),
        in_channels=512,
        out_channels=512,
        kernel_size=3,
        padding=1,
        stride=1,
    )
    results.append(conv27)

    ewadd28 = nn.Ewadd(
        node_id=28,
        name="/layer4/layer4.1/relu_1/Relu_output_0",
        input_shapes=[(1, 512, 7, 7), (1, 512, 7, 7)],
        output_shape=(1, 512, 7, 7),
        activation=nn.ReLU(),
    )
    results.append(ewadd28)

    gavgpool29 = nn.Gavgpool(
        node_id=29,
        name="/avgpool/GlobalAveragePool_output_0",
        input_shapes=[(1, 512, 7, 7)],
        output_shape=(1, 512, 1, 1),
    )
    results.append(gavgpool29)

    conv30 = nn.Conv(
        node_id=30,
        name="output",
        input_shapes=[(1, 512, 1, 1)],
        output_shape=(1, 1000, 1, 1),
        in_channels=512,
        out_channels=1000,
        kernel_size=1,
    )
    results.append(conv30)

    return results


def link_nodes(node_list: List[nn.Node]) -> None:
    node_maps = {
        1: [0],
        2: [1],
        3: [2],
        4: [3, 1],
        5: [4],
        6: [5],
        7: [6, 4],
        8: [7],
        9: [7],
        10: [8],
        11: [10, 9],
        12: [11],
        13: [12],
        14: [13, 11],
        15: [14],
        16: [14],
        17: [15],
        18: [17, 16],
        19: [18],
        20: [19],
        21: [20, 18],
        22: [21],
        23: [21],
        24: [22],
        25: [24, 23],
        26: [25],
        27: [26],
        28: [27, 25],
        29: [28],
        30: [29],
    }

    for child_idx, parent_indexes in node_maps.items():
        extracts = [node_list[i] for i in parent_indexes]
        node_list[child_idx].set_parents(extracts)


if __name__ == "__main__":
    nodes = create_nodes()
    link_nodes(nodes)

    graph = GraphSpearV1()
    graph.set_input_nodes([nodes[0]])
    graph.set_output_nodes([nodes[-1]])
    graph.export("./pyspear-resnet18_b-1.sp")
