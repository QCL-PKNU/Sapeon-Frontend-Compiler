{
  "AIX_INPUT_TYPE": "TENSORFLOW",  
  "AIX_PROFIT_THRESHOLD": 1000,
  "AIX_OPERATION": { 
    "input": ["AIX_LAYER_INPUT", 100],
    "output": ["AIX_LAYER_OUTPUT", 100],
    "Conv2D": ["AIX_LAYER_CONVOLUTION", 100],
    "X_Connect": ["AIX_LAYER_CONNECTED", 100],
    "MaxPool": ["AIX_LAYER_MAXPOOL", 100],
    "AvgPool": ["AIX_LAYER_AVGPOOL", 100],
    "Softmax": ["AIX_LAYER_SOFTMAX", 100],
    "X_Route": ["AIX_LAYER_ROUTE", 100],
    "X_Reorg": ["AIX_LAYER_REORG", 100],
    "Add": ["AIX_LAYER_EWADD", 100],
    "X_Upsample": ["AIX_LAYER_UPSAMPLE", 100],
    "X_PixelShuffle": ["AIX_LAYER_PIXELSHUFFLE", 100],
    "X_SkipConv": ["AIX_LAYER_SKIP_CONV", 100],
    "FusedBatchNorm": ["AIX_LAYER_BATCHNORM", 100],
    "BiasAdd": ["AIX_LAYER_BIASADD", 100],
    "Sigmoid": ["AIX_ACTIVATION_SIGMOID", 100],
    "Relu": ["AIX_ACTIVATION_RELU", 100],
    "LeakyRelu": ["AIX_ACTIVATION_LEAKY_RELU", 100],
    "Prelu": ["AIX_ACTIVATION_PRELU", 100],
    "Tanh": ["AIX_ACTIVATION_TANH", 100],
    "Identity": ["AIX_ACTIVATION_IDENTITY", 100]
  }
}