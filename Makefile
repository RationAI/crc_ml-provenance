PIP=pip3
PYTHON=python3

.PHONY: setup help clean train test

run: setup train test

setup: requirements.txt
	$(PIP) install -r requirements.txt

train: $(TRAIN_CONFIG)
	$(PYTHON) -m rationai.training.experiments.slide_experiment_train $(TRAIN_CONFIG)

test: $(TEST_CONFIG)
	$(PYTHON) -m rationai.training.experiments.slide_experiment_test $(TEST_CONFIG)

help:
	@echo "This is a help string."

clean:
	@echo "Cleaning temporary masks."
