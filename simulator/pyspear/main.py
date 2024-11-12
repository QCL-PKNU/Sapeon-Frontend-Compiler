from pyspear import nn
from pyspear.graph import GraphSpearV1

if __name__ == "__main__":
    act1 = nn.LeakyReLU(0.1)
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
        activation=act1,
    )
    conv1.load_filter_weight(
        "/workspaces/aixgraph_simulator/pyspear/weights.txt"
    )
    conv1.load_filter_bias("/workspaces/aixgraph_simulator/pyspear/bias.txt")

    maxpool = nn.Maxpool(
        node_id=1,
        name="maxpool1",
        input_shapes=[(1, 64, 112, 112)],
        output_shape=(1, 64, 56, 56),
        padding=1,
        window=3,
        stride=2,
    )
    maxpool.set_parents(conv1)

    graph = GraphSpearV1()
    graph.set_input_nodes([conv1])
    graph.set_output_nodes([maxpool])
    graph.export("./pyspear.pb")
