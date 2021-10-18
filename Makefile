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
	@mkdir -p "$(OUTPUT_DIR)/masks/bg/bg_init/"
    @mkdir -p "$(OUTPUT_DIR)/masks/bg/bg_annot/"
    @mkdir -p "$(OUTPUT_DIR)/masks/bg/bg_final/"
    @mkdir -p "$(OUTPUT_DIR)/masks/annotations/"
	$(PYTHON) -m rationai.data.classify.create_map --config_fp $(CONFIG_FILE) --output_dir $(OUTPUT_DIR)
