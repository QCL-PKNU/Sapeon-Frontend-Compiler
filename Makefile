SHELL := /bin/bash

.PHONY: all frontend simulator clean

PYTHON=python3.9

# Frontend
FRONTEND_DIR = frontend
FRONTEND_SRC_DIR = $(FRONTEND_DIR)/src
FRONTEND_SRC = $(FRONTEND_SRC_DIR)/AxfcMain.py
FRONTEND_VENV = $(FRONTEND_DIR)/venv
FRONTEND_REQUIREMENTS = $(FRONTEND_DIR)/requirements.txt

# Simulator
SIMULATOR_DIR = simulator
SIMULATOR_SRC = $(SIMULATOR_DIR)/main.cpp $(SIMULATOR_DIR)/simulator.cpp

all: frontend simulator

frontend: $(FRONTEND_VENV)
	@echo "Install Python dependencies..."
	source $(FRONTEND_VENV)/bin/activate && pip install -r $(FRONTEND_REQUIREMENTS)

	@echo "Compile deep learning model...$(MODEL_PATH)"
	source $(FRONTEND_VENV)/bin/activate && $(PYTHON) $(FRONTEND_SRC)  -i $(FRONTEND_DIR)/assets/models/$(MODEL_NAME) \
																    -c $(FRONTEND_DIR)/assets/calibs/$(CALIB_FILE) \
																    -m $(FRONTEND_DIR)/assets/md/$(MD_FILE) \
																	-l $(FRONTEND_DIR)/assets/logging.log \
																	-o $(FRONTEND_DIR)/assets/aix_graph.out \
																	-f $(GRAPH_FORMAT) \
																	$(if $(INPUT_SHAPE),-s $(INPUT_SHAPE))\

	@echo "Model has been compile successfully"

$(FRONTEND_VENV):
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(FRONTEND_VENV)


simulator: simulator_env
	@echo "Copying all 'aix_graph.out*' files to simulator/assets..."
	mkdir -p $(SIMULATOR_DIR)/assets
	cp $(FRONTEND_DIR)/assets/aix_graph.out* $(SIMULATOR_DIR)/assets/

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
