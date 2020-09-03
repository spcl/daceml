VENV_PATH ?= venv
PYTHON ?= python
PYTEST ?= pytest
PIP ?= pip
YAPF ?= yapf

clean:
	! test -d $(VENV_PATH) || rm -r $(VENV_PATH)

reinstall: venv
ifeq ($(VENV_PATH),)
	$(PIP) install --upgrade --force-reinstall -e .[testing] 
else
	. $(VENV_PATH)/bin/activate && $(PIP) install --upgrade --force-reinstall -e .[testing] 
endif


venv: 
ifneq ($(VENV_PATH),)
	test -d $(VENV_PATH) || echo "Creating new venv" && python3 -m venv ./$(VENV_PATH)
	ls $(VENV_PATH)/bin
endif

install: venv
ifeq ($(VENV_PATH),)
	$(PIP) install -e .[testing]
else
	. $(VENV_PATH)/bin/activate && $(PIP) install -e .[testing] 
endif

test: 
ifeq ($(VENV_PATH),)
	$(PYTEST) tests
else
	. $(VENV_PATH)/bin/activate && $(PYTEST) tests
endif

test-gpu: 
ifeq ($(VENV_PATH),)
	$(PYTEST) tests --gpu
else
	. $(VENV_PATH)/bin/activate && $(PYTEST) tests --gpu
endif

check-formatting:
ifeq ($(VENV_PATH),)
	$(YAPF) --parallel \
		--diff \
		--recursive \
		daceml tests setup.py \
	       	--exclude daceml/onnx/symbolic_shape_infer.py
else
	. $(VENV_PATH)/bin/activate && $(YAPF) --parallel \
		--diff \
		--recursive \
		daceml tests setup.py \
		--exclude daceml/onnx/symbolic_shape_infer.py
endif
