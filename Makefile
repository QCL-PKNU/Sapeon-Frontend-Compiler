######## PATH CONFIG ########

MODEL= ./tst/resnet50_v1.pb
MD= ./tst/resnet50_v1_aix_tf.md
AXFC= ./tst/axfc_data.json
LOGGING= ./tst/logging.log
CALIB= ./tst/resnet50_v1_imagenet_calib.tbl
KERNEL= ./tst/custom_op_kernel.so

all: 
	python3 -tt src/AxfcMain.py -m $(MD) -i $(MODEL) -g $(AXFC) -l $(LOGGING) -c $(CALIB) -k $(KERNEL)

clean:
	rm -rf tst/aix_graph.out.00
	rm -rf tst/custom_model.pb
	rm -rf tst/custom_op_kernel.so
