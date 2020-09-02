VENV_PATH ?= venv
PYTHON ?= $(VENV_PATH)/bin/python
PYTEST ?= $(VENV_PATH)/bin/pytest
PIP ?= $(VENV_PATH)/bin/pip

clean:
	rm -r $(VENV_PATH)

reinstall:
	$(PIP) install --upgrade --force-reinstall -e .[testing] 

install:
	python3 -m venv $(VENV_PATH)
	$(PIP) install -e .[testing]
	. $(VENV_PATH)/bin/activate

test:
	$(PYTEST) tests
