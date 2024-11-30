{
	"AIX_MODEL_TYPE": "TENSORFLOW", 
	"AIX_MODEL_NAME": "resnet_model", 
	"AIX_PROFIT_THRESHOLD": 500,
	"STOP_COMPILING_POINT":"",
	"AIX_LAYER": { 
		"Conv2D": {
			"layer": "AIX_LAYER_CONVOLUTION",
			"activation": "AIX_ACTIVATION_IDENTITY",
			"is_group": false,
			"is_conv": true,
			"profit": 100
		},
		"DepthwiseConv2dNative": {
			"layer": "AIX_LAYER_GROUP_CONV",
			"activation": "AIX_ACTIVATION_IDENTITY",
			"is_group": true,
			"is_conv": false,
			"profit": 100
		},
		"FusedBatchNormV3": {
			"layer": "AIX_LAYER_BATCHNORM",
			"activation": null,
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},
		"FusedBatchNorm": {
			"layer": "AIX_LAYER_BATCHNORM",
			"activation": null,
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},
		"BatchNorm": {
			"layer": "AIX_LAYER_BATCHNORM",
			"activation": null,
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},
		"AvgPool": {
			"layer": "AIX_LAYER_AVGPOOL",
			"activation": null,
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},
		"MaxPool": {
			"layer": "AIX_LAYER_MAXPOOL",
			"activation": null,
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},      
		"Softmax": {
			"layer": "AIX_LAYER_SOFTMAX",
			"activation": null,
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},
		"AddV3": {
			"layer": "AIX_LAYER_EWADD",
			"activation": null,
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},
		"AddV2": {
			"layer": "AIX_LAYER_EWADD",
			"activation": null,
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},                        
		"Add": {
			"layer": "AIX_LAYER_EWADD",
			"activation": null,
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},           
		"Relu": {
			"layer": "AIX_LAYER_ACTIVATION",
			"activation": "AIX_ACTIVATION_RELU",
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},
		"Relu6": {
			"layer": "AIX_LAYER_ACTIVATION",
			"activation": "AIX_ACTIVATION_RELU",
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},
		"BiasAdd": {
			"layer": "AIX_LAYER_BIASADD",
			"activation": null,
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},      
		"Sigmoid": {
			"layer": "AIX_LAYER_ACTIVATION",
			"activation": "AIX_ACTIVATION_SIGMOID",
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},        
		"Prelu": {
			"layer": "AIX_LAYER_ACTIVATION",
			"activation": "AIX_ACTIVATION_PRELU",
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},   
		"Tanh": {
			"layer": "AIX_LAYER_ACTIVATION",
			"activation": "AIX_ACTIVATION_TANH",
			"is_group": false,
			"is_conv": false,
			"profit": 100
		}	
	}
}