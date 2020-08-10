{
	"AIX_MODEL_TYPE": "TENSORFLOW",  
	"AIX_PROFIT_THRESHOLD": 1000,
	"AIX_LAYER": { 
		"Conv2D": {
			"layer": "AIX_LAYER_CONVOLUTION",
			"inputs": {
			    "filter": "weights"
			},
			"activation": "AIX_ACTIVATION_IDENTITY",
			"is_group": false,
			"is_conv": true,
			"profit": 100
		},
		"DepthwiseConv2dNative": {
			"layer": "AIX_LAYER_CONVOLUTION",
			"inputs": {
			    "filter": "depthwise_weights"
			},			
			"activation": null,
			"is_group": true,
			"is_conv": true,
			"profit": 100
		},
		"FusedBatchNorm": {
			"layer": "AIX_LAYER_BATCHNORM",
			"inputs": {
			    "scale": "gamma",
			    "offset": "beta",
			    "mean": "moving_mean",
			    "variance": "moving_variance"
			},				
			"activation": null,
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},
		"BatchNorm": {
			"layer": "AIX_LAYER_BATCHNORM",
			"inputs": {
			    "scale": "gamma",
			    "offset": "beta",
			    "mean": "moving_mean",
			    "variance": "moving_variance"
			},		
			"activation": null,
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},
		"AvgPool": {
			"layer": "AIX_LAYER_AVGPOOL",
			"inputs": null,
			"activation": null,
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},
		"MaxPool": {
			"layer": "AIX_LAYER_MAXPOOL",
			"inputs": null,
			"activation": null,
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},      
		"Softmax": {
			"layer": "AIX_LAYER_SOFTMAX",
			"inputs": null,
			"activation": null,
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},      
		"Add": {
			"layer": "AIX_LAYER_EWADD",
			"inputs": null,
			"activation": null,
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},           
		"Relu6": {
			"layer": "AIX_LAYER_ACTIVATION",
			"inputs": null,
			"activation": "AIX_ACTIVATION_LEAKY_RELU",
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},
		"BiasAdd": {
			"layer": "AIX_LAYER_ACTIVATION",
			"inputs": null,
			"activation": "AIX_LAYER_BIASADD",
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},      
		"Sigmoid": {
			"layer": "AIX_LAYER_ACTIVATION",
			"inputs": null,
			"activation": "AIX_ACTIVATION_SIGMOID",
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},        
		"Prelu": {
			"layer": "AIX_LAYER_ACTIVATION",
			"inputs": null,
			"activation": "AIX_ACTIVATION_PRELU",
			"is_group": false,
			"is_conv": false,
			"profit": 100
		},   
		"Tanh": {
			"layer": "AIX_LAYER_ACTIVATION",
			"inputs": null,
			"activation": "AIX_ACTIVATION_TANH",
			"is_group": false,
			"is_conv": false,
			"profit": 100
		}     
	}
}