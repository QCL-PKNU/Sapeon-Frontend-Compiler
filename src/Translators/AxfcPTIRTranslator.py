from .AxfcIRTranslator import *

class AxfcPTIRTranslator(AxfcIRTranslator):

    ## The constructor
    def __init__(self, md, path):
        super().__init__(md)

    ## emission methods for AIX layers
    def _emit_aix_layer_convolution(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_group_conv(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_batchnorm(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_maxpool(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_ewadd(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_avgpool(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_biasadd(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_softmax(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        return NotImplementedError()

    def _emit_aix_layer_activation(self, ir_node: AxfcIRNode, **kwargs) -> AxfcError:
        return NotImplementedError()

    ## emission methods for AIX tensors
    def _emit_aix_tensor_input(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    def _emit_aix_tensor_filter(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    def _emit_aix_tensor_bias(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    def _emit_aix_tensor_scale(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    def _emit_aix_tensor_mean(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    def _emit_aix_tensor_variance(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    def _emit_aix_tensor_output(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    ## emission methods for AIX convolution dec
    def _emit_aix_convolution_desc(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        return NotImplementedError()

    ## emission methods for AIX sampling dec
    def _emit_aix_sampling_desc(self, ir_node: AxfcIRNode, **kwargs) -> AIXLayer.AIXTensor:
        return NotImplementedError()