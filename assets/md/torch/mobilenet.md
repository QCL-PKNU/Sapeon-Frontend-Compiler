{
"AIX_MODEL_TYPE": "PYTORCH",
"AIX_MODEL_NAME": "resnet_model",
"AIX_PROFIT_THRESHOLD": 500,
"STOP_COMPILING_POINT":"",
"AIX_LAYER": {
"Conv2d": {
"layer": "AIX_LAYER_CONVOLUTION",
"activation": "AIX_ACTIVATION_IDENTITY",
"is_group": false,
"is_conv": true,
"profit": 100
},
"BatchNorm2d": {
"layer": "AIX_LAYER_BATCHNORM",
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
"ReLU6":{
"layer": "AIX_LAYER_ACTIVATION",
"activation": "AIX_ACTIVATION_RELU",
"is_group": false,
"is_conv": false,
"profit": 100
},
"MaxPool2d":{
"layer": "AIX_LAYER_MAXPOOL",
"activation": null,
"is_group": false,
"is_conv": false,
"profit": 100
},
"AvgPool2d":{
"layer": "AIX_LAYER_AVGPOOL",
"activation": null,
"is_group": false,
"is_conv": false,
"profit": 100
}
}
}
