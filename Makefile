SHELL := /bin/bash

.PHONY: all frontend clean

PYTHON=python3.9

all: frontend

frontend: $(FRONTEND_VENV)
	@echo "Install Python dependencies..."
	source venv/bin/activate && pip install -r requrements.txt

	@echo "Compile deep learning model...$(MODEL_PATH)"
	source venv/bin/activate && $(PYTHON) src/AxfcMain.py \  
																	-i assets/models/$(MODEL_NAME) \
																  -c assets/calibs/$(CALIB_FILE) \
																  -m assets/md/$(MD_FILE) \
																	-l assets/logging.log \
																	-o assets/aix_graph.out \
																	-f $(GRAPH_FORMAT) \
																	$(if $(INPUT_SHAPE),-s $(INPUT_SHAPE))\

	@echo "Model has been compile successfully"

$(FRONTEND_VENV):
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv venv
	

clean:
	@echo "Clearning build files..."
	rm -rf venv
	rm -f logs/*.log