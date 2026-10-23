PYTHON ?= python3

.PHONY: all run verify install clean

all: run verify

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) code/run.py

verify:
	$(PYTHON) code/verify.py

clean:
	rm -rf output code/__pycache__
