SHELL := /bin/bash

.PHONY: all frontend simulator clean

FRONTEND_DIR = frontend
FRONTEND_SRC_DIR = $(FRONTEND_DIR)/src
FRONTEND_SRC = $(FRONTEND_SRC_DIR)/AxfcMain.py
FRONTEND_VENV = $(FRONTEND_DIR)/venv
FRONTEND_REQUIREMENTS = $(FRONTEND_DIR)/requirements.txt

ifeq ($(MODEL_TYPE), onnx)
    MODEL_PATH = $(FRONTEND_DIR)/assets/models/resnet50-v1-7.onnx
    CALIB_PATH = $(FRONTEND_DIR)/assets/calibs/resnet50_v1_imagenet_calib.tbl
    MACHINE_DESC = $(FRONTEND_DIR)/assets/md/onnx_sample.md
	GRAPH_FORMAT = binary
else ifeq ($(MODEL_TYPE), pytorch)
    MODEL_PATH = $(FRONTEND_DIR)/assets/models/resnet50.pt
    CALIB_PATH = $(FRONTEND_DIR)/assets/calibs/resnet50_v1_imagenet_calib.tbl
    MACHINE_DESC = $(FRONTEND_DIR)/assets/md/torch_sample.md
	GRAPH_FORMAT = text
else
    # Default to TensorFlow
    MODEL_PATH = $(FRONTEND_DIR)/assets/models/resnet50.pb
    CALIB_PATH = $(FRONTEND_DIR)/assets/calibs/resnet50_v1_imagenet_calib.tbl
    MACHINE_DESC = $(FRONTEND_DIR)/assets/md/tensorflow_sample.md
	GRAPH_FORMAT = binary
endif

LOGGING_PATH = $(FRONTEND_DIR)/assets/logging.log
GRAPH_OUT_PATH = $(FRONTEND_DIR)/assets/aix_graph.out

SIMULATOR_DIR = simulator
SIMULATOR_SRC = $(SIMULATOR_DIR)/main.cpp $(SIMULATOR_DIR)/simulator.cpp


all: frontend simulator

frontend: $(FRONTEND_VENV)
	@echo "Install Python dependencies..."
	source $(FRONTEND_VENV)/bin/activate && pip install -r $(FRONTEND_REQUIREMENTS)

	@echo "Compile deep learning model...$(MODEL_PATH)"
	source $(FRONTEND_VENV)/bin/activate && python $(FRONTEND_SRC)  -i $(MODEL_PATH) \
																    -c $(CALIB_PATH) \
																    -m $(MACHINE_DESC) \
																	-l $(LOGGING_PATH) \
																	-o $(GRAPH_OUT_PATH) \
																	-f $(GRAPH_FORMAT) \

	@echo "Copying all 'aix_graph.out*' files to simulator/assets..."
	mkdir -p $(SIMULATOR_DIR)/assets
	cp $(FRONTEND_DIR)/assets/aix_graph.out* $(SIMULATOR_DIR)/assets/


$(FRONTEND_VENV):
	@echo "Creating virtual environment..."
	python3 -m venv $(FRONTEND_VENV)


simulator: simulator_env
	@echo "Compile and run the AIXGraph Simulator..."
	cd $(SIMULATOR_DIR) && ./simulator --backend cpu \
											--model-path assets/aix_graph.out.0.pb \
											--graph-type aix_graph \
											--infer \
											--image-path assets/cat.jpg \
											--dump-dir outputs

simulator_env:
	@echo "Create environment for simulator"
	mkdir -p $(SIMULATOR_DIR)/build
	cd $(SIMULATOR_DIR)/build && cmake .. && make

clean:
	@echo "Clearning build files..."
	rm -rf $(FRONTEND_DIR)/venv
	rm -f $(FRONTEND_DIR)/logs/*.log
	rm -rf $(SIMULATOR_DIR)/simulator
	rm -rf $(SIMULATOR_DIR)/build

