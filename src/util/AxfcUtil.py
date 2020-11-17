#######################################################################
#   AxfcUtil
#
#   Created: 2020. 09. 10
#
#   Authors:
#      Youngsun Han (youngsun@pknu.ac.kr)
#      Heng Sengthai (sengthai37@gmail.com)
#
#   High Performance Computing Laboratory (hpcl.pknu.ac.kr)
#######################################################################

import tensorflow as tf


#######################################################################
# AxfcUtil methods
#######################################################################


## This method is used to load the frozen model to tensorflow graphdef object
#
# @param file_name the path of tensorflow frozen model
# @return graph_def the object of tf.graph_pb2.GraphDef
def loadFrozenModel(file_name):
    db_path = file_name
    f = tf.io.gfile.GFile(db_path, 'rb')
    f_byte = f.read()
    f.close()
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f_byte)
    return graph_def


## This method is used to write the graphdef to file (.pbtxt)
#
# @param graph_def the object of tf.graph_db2.GraphDef
# @param des_path the destination path for storing file 
# @param name_file the file name
# @return path the path of file
def write2pbtxt(graph_def, des_path, name_file):
    return tf.io.write_graph(graph_def, des_path, "{}.pbtxt".format(name_file))


## This method is used to write the graphdef to protobuf file (.pb)
#
# @param graph_def the object of tf.graph_db2.GraphDef
# @param des_path the destination path for storing file 
# @param name_file the file name
# @return path the path of file
def write2pb(graph_def, des_path, name_file):
    return tf.io.write_graph(graph_def, des_path, "{}.pb".format(name_file), as_text=False)


## this method is used to print the node name from graphdef
#
def print_name(graph_def):
    for node in graph_def.node:
        print(node.name)


## This method is used to print tensor as graph def
#
# @param tensor the tensor object
# @return graph_def the object of tf.graph_db2.GraphDef
def print_def(tensor):
    return tensor.graph.as_graph_def()


## This method is used to print out the encoded value of tensor content in layer
#
# @param operation the operation object
# @return value the decoded value of tensor content
def print_tensor_content(operation):
    if 'value' in operation.node_def.attr:
        return tf.make_ndarray(operation.get_attr('value'))
    else:
        return None
