PIP=pip3
PYTHON=python3

.PHONY: setup help clean run

setup: requirements.txt
	$(PIP) install -r requirements.txt

help:
	@echo "This is a help string for create_map."

clean:
	@echo "Cleaning temporary masks."

run:
	$(PYTHON) -m rationai.data.classify.create_map --config_fp $(CONFIG_FILE)
