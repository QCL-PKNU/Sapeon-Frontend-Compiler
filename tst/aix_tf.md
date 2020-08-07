{
	"AIX_MODEL_TYPE": "TENSORFLOW",  
	"AIX_PROFIT_THRESHOLD": 1000,
	"AIX_OPERATION": { 
		"Conv2D": {
			"layer": "AIX_LAYER_CONVOLUTION",
			"activation": null,
			"is_group": false,
			"is_conv": true,
			"profit": 100
		},
		"DepthwiseConv2dNative": {
			"layer": "AIX_LAYER_CONVOLUTION",
			"activation": null,
			"is_group": true,
			"is_conv": true,
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
		"Add": {
			"layer": "AIX_LAYER_EWADD",
			"activation": null,
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},           
		"Relu6": {
			"layer": "AIX_ACTIVATION",
			"activation": "AIX_ACTIVATION_LEAKY_RELU",
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},
		"BiasAdd": {
			"layer": "AIX_ACTIVATION",
			"activation": "AIX_LAYER_BIASADD",
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},      
		"Sigmoid": {
			"layer": "AIX_ACTIVATION",
			"activation": "AIX_ACTIVATION_SIGMOID",
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},        
		"Prelu": {
			"layer": "AIX_ACTIVATION",
			"activation": "AIX_ACTIVATION_PRELU",
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},   
		"Tanh": {
			"layer": "AIX_ACTIVATION",
			"activation": "AIX_ACTIVATION_TANH",
			"is_group": false,
			"is_conv": false,
			"profit": 100
		}     
	}
}