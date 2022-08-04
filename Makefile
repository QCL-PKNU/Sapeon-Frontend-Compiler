######## MODEL PATH CONFIG ########
MODEL= ./tst/retinanet.pb
MD= ./tst/retinanet_v1_aix_tf.md
AIX_GRAPH_FORMAT= BINARY
# KERNEL= ./tst/custom_op_kernel.so

######## LOG PATH CONFIG ########
CALIB= ./tst/resnet50_v1_imagenet_calib.tbl
AXFC= ./tst/axfc_data.json
LOGGING= ./tst/logging.log

all:
	python3 -tt src/AxfcMain.py -m $(MD) -i $(MODEL) -g $(AXFC) -l $(LOGGING) -c $(CALIB) -f $(AIX_GRAPH_FORMAT)

clean:
	rm -rf tst/aix_graph.out*
