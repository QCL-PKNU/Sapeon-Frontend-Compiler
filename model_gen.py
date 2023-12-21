import hiddenlayer as hl
import torch
import torch.onnx
import onnx
import numpy as np
import onnx.numpy_helper as numpy_helper

from onnx            import shape_inference
from torchvision     import models
from torch.autograd  import Variable
import torch.fx 


#Load the resnet50 resnet50
resnet50 = models.resnet50(weights='DEFAULT')


def my_randn(*args, **kwargs):
    return torch.randn(*args, **kwargs)

#Make dummy input_data
# input_data = Variable(torch.randn(1, 3, 224, 224))
input_data = my_randn((1, 3, 244, 244))

#Resnet model
resnet50_model = 'resnet50.pt'

#Torch prm
resnet50_state_file = 'resnet50_state.pth'

#ONNX output
onnx_file = 'resnet50_p2o.onnx'


#Function for freezing all parameters of resnet50 model
def freezeMD():
    # print(resnet50)

    #Freeze all parameters
    for param in resnet50.parameters():
        param.requires_grad = False

    for name, module in resnet50.named_modules():
        for _, param in module.named_parameters():
            param.requires_grad = True

    #Save the resnet50 with parameter
    params = resnet50.state_dict()
    
    # print(resnet50)
    # print(params)

    #Save the pth
    torch.save(resnet50, resnet50_model, pickle_protocol=4)
    
    #Save the state file
    torch.save(params, resnet50_state_file, pickle_protocol=4)


def do_script():
    pt_model = torch.load("/home/sanghyeon/repos/aix-project/skt-aix-frontend-compiler/tst/resnet50.pt")

    pt_model.load_state_dict(torch.load("/home/sanghyeon/repos/aix-project/skt-aix-frontend-compiler/resnet50_state.pth"))

    pt_model.eval()

    #Freeze all parameters
    scripted_model = torch.jit.trace(pt_model, input_data)

    scripted_model.save("scripted_model.pt")




#This function is for visualize the original PyTorch model graph
def visualize():
    transform = [hl.transforms.Prune('Constant')]

    #Visualize the graph
    graph = hl.build_graph(resnet50, input_data, transforms=transform)

    #Set the theme
    graph.theme = hl.graph.THEMES['blue'].copy()

    #Save as pdf format
    graph.save('resnet50_pt', format='pdf')


#Function to convert the PyTorch into ONNX model
def pt2onnx():
    #Load the prm
    prm = torch.load(f'./{resnet50_state_file}')

    resnet50.load_state_dict(prm)

    #Set resnet50 as evaluation mode
    # resnet50.eval()

    #Save in Training mode    
    torch.onnx.export(resnet50, input_data, onnx_file, training=torch.onnx.TrainingMode.TRAINING)

    path = f'./{onnx_file}'

    #Add the shape information to onnx graph
    onnx.save(shape_inference.infer_shapes(onnx.load(path)), path)

###############################
# Debugging
###############################
'''
Below from here.
This code is to check the onnx model which has the different weight compared to the PyTorch model.
Because, the onnx store the model in different way. It unpack the weight, bias value from original model.
    
    1.Load the onnx and pytorch model.
    2.Save its layer's information.
    3.Compare each layer's name and weight between onnx and torch model.
    4.Update the layer name and weight of onnx model which has the different one.

Refer) https://gaussian37.github.io/dl-pytorch-deploy/
'''
# # #Function to compare the layer
# def compare_two_array(actual, desired, layer_name, rtol=1e-7, atol=0):
#     flag = False
#     try : 
#         np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)
#         print(layer_name + ": no difference.")
#     except AssertionError as msg:
#         print(layer_name + ": Error.")
#         print(msg)
#         flag = True
#     return flag

# #Load the generated onnx model
# onnx_model = onnx.load(f'./{onnx_file}')

# #Store onnx layer information
# onnx_layers = dict()
# for layer in onnx_model.graph.initializer:
#     onnx_layers[layer.name] = numpy_helper.to_array(layer)

# #Store Torch layer information
# torch_layers = dict()
# for layer_name, layer_value in resnet50.named_modules():
#     torch_layers[layer_name] = layer_value

# #Make the onnx layer set
# onnx_layers_set = set(onnx_layers.keys())

# #Make the torch layer set
# torch_layers_set = set([layer_name + '.weight' for layer_name in list(torch_layers.keys())])

# #Find the intersected layer
# filtered_onnx_layers = list(onnx_layers_set.intersection(torch_layers_set))

# #Flag bit
# difference_flag = False

# #Find the matched layer
# for layer_name in filtered_onnx_layers:
#     #Load the layer name
#     onnx_layer_name = layer_name
#     torch_layer_name = layer_name.replace('.weight', "")
    
#     #Load the weight
#     onnx_weight = onnx_layers[onnx_layer_name]
#     torch_weight = torch_layers[torch_layer_name].weight.detach().numpy()
    
#     #Compare
#     flag = compare_two_array(onnx_weight, torch_weight, onnx_layer_name)
#     difference_flag = True if flag == True else False

# #Replace
# if difference_flag:
#     print("Update onnx weight from torch model")
#     for index, layer in enumerate(onnx_model.graph.initializer):
#         layer_name = layer.name
#         if layer_name in filtered_onnx_layers:
#             #Replace layer name
#             onnx_layer_name = layer_name
#             torch_layer_name = layer_name.replace(".weight", "")
            
#             #Replace layer weight
#             onnx_weight = onnx_layers[onnx_layer_name]
#             torch_weight = torch_layers[torch_layer_name].weight.detach().numpy()
            
#             copy_tensor = numpy_helper.from_array(torch_weight, onnx_layer_name)
#             onnx_model.graph.initializer[index].CopyFrom(copy_tensor)
    
#     print('save update onnx model')    
    
#     #Save updated model
#     onnx.save(shape_inference.infer_shapes(onnx_model), 'updated_resnet_p2o.onnx')


if __name__ == '__main__':
    freezeMD()

    do_script()

    # visualize()