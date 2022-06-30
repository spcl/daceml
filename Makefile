VENV_PATH ?= venv
PYTHON ?= python
PYTHON_BINARY ?= python
PYTEST ?= pytest
PIP ?= pip
YAPF ?= yapf

TORCH_VERSION ?= torch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1
DACE_VERSION ?=
UPDATE_PIP ?= python -m pip install --upgrade pip
SOURCE_FILES = daceml tests setup.py examples doc

ifeq ($(VENV_PATH),)
ACTIVATE = 
else
ACTIVATE = . $(VENV_PATH)/bin/activate &&
endif

.PHONY: clean doc doctest test test-gpu codecov check-formatting check-formatting-names clean-dacecaches yapf
clean:
	! test -d $(VENV_PATH) || rm -r $(VENV_PATH)

venv: 
ifneq ($(VENV_PATH),)
	test -d $(VENV_PATH) || echo "Creating new venv" && $(PYTHON) -m venv ./$(VENV_PATH)
endif

install: venv
ifneq ($(VENV_PATH),)
	$(ACTIVATE) $(UPDATE_PIP)
endif
ifneq ($(DACE_VERSION),)
	$(ACTIVATE) $(PIP) install $(DACE_VERSION)
endif
	$(ACTIVATE) $(PIP) install $(TORCH_VERSION)
	$(ACTIVATE) $(PIP) install -e .[testing,docs]

doc:
# suppress warnings in ONNXOps docstrings using grep -v
	$(ACTIVATE) cd doc && make clean html 2>&1 \
	| grep -v ".*daceml\/daceml\/onnx\/nodes\/onnx_op\.py:docstring of daceml\.onnx\.nodes\.onnx_op\.ONNX.*:[0-9]*: WARNING:"


doctest:
	$(ACTIVATE) cd doc && make doctest

test: 
	$(ACTIVATE) $(PYTEST) $(PYTEST_ARGS) tests

test-parallel: 
	$(ACTIVATE) $(PYTEST) $(PYTEST_ARGS) tests -n auto --dist loadfile

test-gpu: 
	$(ACTIVATE) $(PYTEST) $(PYTEST_ARGS) tests --gpu

test-intel-fpga:
	$(ACTIVATE) $(PYTEST) $(PYTEST_ARGS) tests/pytorch/fpga/

test-xilinx:
	$(ACTIVATE) $(PYTEST) $(PYTEST_ARGS) tests/pytorch/fpga/

codecov:
	curl -s https://codecov.io/bash | bash

clean-dacecaches:
	find . -name ".dacecache" -type d -not -path "*CMakeFiles*" -exec rm -r {} \; || true

check-formatting:
	$(ACTIVATE) $(YAPF) \
		--parallel \
		--diff \
		--recursive \
		$(SOURCE_FILES) \
		--exclude daceml/onnx/shape_inference/symbolic_shape_infer.py
	# check for sdfg.view()
	! git grep '\.view()' -- tests/** daceml/**

format:
	@$(ACTIVATE) \
		DIFF=$$($(YAPF) \
		--parallel \
		--diff \
		--recursive \
		$(SOURCE_FILES) \
		--exclude daceml/onnx/shape_inference/symbolic_shape_infer.py); \
		if [ -z "$$DIFF" ]; then \
			echo "All files formatted correctly"; \
			exit 0; \
		fi; \
		FILES=$$(echo "$$DIFF" | grep -oP '\+\+\+\s+\K.*(?=\s+\(reformatted\))') \
		&& echo "Going to format:\n$$FILES" && echo -n "Ok? [y/N]" \
		&& read ans && [ $${ans:-N} = y ] && echo "Formatting..." \
		&& echo "$$FILES" | xargs $(YAPF) --parallel --in-place
