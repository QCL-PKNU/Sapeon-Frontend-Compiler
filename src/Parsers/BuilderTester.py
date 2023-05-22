import AxfcIRBuilder
from AxfcONNXIRBuilder import *
from AxfcTFIRBuilder import *
from AxfcPTBuilder import *

def testing():
    onnx_builder = AxfcONNXIRBuilder(AxfcIRBuilder)

    # tf_builder = AxfcTFIRBuilder(AxfcIRBuilder)

    pt_builder = AxfcPTBuilder(AxfcIRBuilder)

    # onnx_builder._read_model_graph("/home/sanghyeon/repos/aix-project/skt-aix-frontend-compiler/tst/mobilenetv2-7.onnx")
    # onnx_builder._build_naive_ir("/home/sanghyeon/repos/aix-project/skt-aix-frontend-compiler/tst/mobilenetv2-7.onnx")
    
    pt_builder._read_model_graph("/home/sanghyeon/repos/aix-project/skt-aix-frontend-compiler/tst/resnet50.pt")

    # tf_builder._read_model_graph("/home/sanghyeon/repos/aix-project/skt-aix-frontend-compiler/tst/resnet50_v1.pb")

    # print(pt_builder)
    # print(tf_builder.__tf_graph)

if __name__ == '__main__':
    testing()
