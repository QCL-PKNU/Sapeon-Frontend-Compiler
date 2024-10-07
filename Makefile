# Makefile for Python project with virtual environment setup

# Variables
PYTHON = ./venv/bin/python3
PIP = ./venv/bin/pip3

MODEL_PATH = tst/resnet50.pb
MD_PATH = tst/tf_sample.md
CALIB_PATH = tst/resnet50_v1_imagenet_calib.tbl
LOGGING_PATH = tst/logging.log
FORMAT = binary

# Model type selection (uncomment the relevant model block)
# Uncomment below block for ONNX model
# MODEL_PATH = tst/resnet50-v1-7.onnx
# MD_PATH = tst/onnx_sample.md
# CALIB_PATH = tst/mobilenet_calib.tbl

# Uncomment below block for PyTorch model
# MODEL_PATH = tst/resnet50.pt
# MD_PATH = tst/torch_sample.md
# CALIB_PATH = tst/resnet50_v1_imagenet_calib.tbl

# Virtual environment directory
VENV_DIR = ./venv

# Ensure the virtual environment is created
$(VENV_DIR)/bin/activate: requirements.txt
	@echo "Setting up virtual environment..."
	python3 -m venv $(VENV_DIR)
	@echo "Upgrading pip and installing dependencies..."
	$(VENV_DIR)/bin/pip3 install --upgrade pip
	$(VENV_DIR)/bin/pip3 install -r requirements.txt
	@echo "Virtual environment setup complete."

# Create or activate the virtual environment
venv: $(VENV_DIR)/bin/activate
	@echo "Virtual environment ready. To activate, use 'source $(VENV_DIR)/bin/activate'."

# Run the main script with the specified parameters
all: venv
	@echo "Running the main application..."
	$(PYTHON) -tt src/AxfcMain.py \
		-m $(MD_PATH) \
		-i $(MODEL_PATH) \
		-l $(LOGGING_PATH) \
		-c $(CALIB_PATH) \
		-f $(FORMAT)

# Remove virtual environment, cached files, and generated outputs
clean:
	@echo "Cleaning up build artifacts and virtual environment..."
	rm -rf __pycache__ $(VENV_DIR) tst/aix_graph.out.*
	@echo "Cleanup complete."

# Ensure certain targets are not associated with actual file names
.PHONY: venv run clean
