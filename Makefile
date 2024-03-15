.ONESHELL:

###### PATH CONFIG ######

### Tensorflow framework and header path
TF_CFLAGS := $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
##########################

DEBUG = 1
GCC = g++
LOGGING= ./tst/logging.log
AXFC= ./tst/axfc_data.json
LOGGING= ./tst/logging.log
CALIB= ./tst/resnet50_v1_imagenet_calib.tbl
PYTHON = ./venv/bin/python3
PIP = ./venv/bin/pip3
DARKNET = ./darknet_mxconv
CUSTOM = ./custom
MODEL_TYPE = 2
ASSETS = assets
AIX_GRAPH_FORMAT = text

GCCFLASS = -std=c++11 -Wall -fpermissive -fPIC -msse4.1
PLDA_LIB = $(DARKNET)/library/driver/aixasic/plda_drv/qpcie_api/bin/x86_64/libpldaqpcie_lib.a
LDFLAGS  = -L$(DARKNET)/ -L$(DARKNET)/library/ -L/usr/local/lib/ -lm -pthread -lrt -lglog  \-laix -ldl -lnvp_uicore
LDFLAGS += $(PLDA_LIB) /usr/lib/x86_64-linux-gnu/librt.so
LDFLAGS += `pkg-config --cflags --libs protobuf`
LDFLAGS += -L$(DARKNET)/library/driver/fs/aixh/build/ -lfs -ldl -lm -pthread -lglog
COMMON = -I$(DARKNET)/include/ -I$(DARKNET)/src/ -I$(DARKNET)/library -I$(DARKNET)/library/highlevelAPI/include

venv/bin/activate: requirements.txt
	python3 -m venv venv
	chmod +x venv/bin/activate
	. ./venv/bin/activate
	$(PIP) install -r requirements.txt

venv: venv/bin/activate
	. ./venv/bin/activate


ALIB=$(DARKNET)/libdarknet.a

ifeq ($(DEBUG), 1) 
OPTS=-O0 -ggdb3 -g3
endif

ifeq ($(MODEL_TYPE), 1)
	MODEL= $(ASSETS)/mobilenet_v1_1.0_224_frozen.pb
	MD= $(ASSETS)/retinanet_v1_aix_tf.md
else
	MODEL = tst/resnet50.pt
	MD= tst/torch_sample.md
endif


all: makefile custom_op_kernel test

custom_op_kernel:
	$(GCC) $(COMMON) -g -z defs -shared $(CUSTOM)/custom_op_kernel.cc -o $(custom)/custom_op_kernel.so $(LDFLAGS) $(LDFLAGS) $(TF_CFLAGS)

test:
	$(GCC) $(COMMON) $(CCCFLAGS) $(OPTS) $(CUSTOM)/test_main.cc -o ./src/test_main $(LDFLAGS) $(ALIB) 

makefile: makefile_libra
	$(MAKE) -C $(DARKNET)/ all PROFILE=asic_128x64

makefile_library:
	$(MAKE) -C $(DARKNET)/library/ all PROFILE=asic_128x64

run: venv
	$(PYTHON) -tt src/AxfcMain.py -m $(MD) -i $(MODEL) -g $(AXFC) -l $(LOGGING) -c $(CALIB) -f $(AIX_GRAPH_FORMAT)

clean: clean_kernel

clean_kernel:
	$(MAKE) -C $(DARKNET)/ clean
	$(MAKE) -C $(DARKNET)/library/ clean
	rm -rf $(CUSTOM)/custom_op_kernel.so

clean_file:
	rm -rf __pycache__
	rm -rf venv
	rm -rf tst/aix_graph.out*