# If experiment directory has not been provided, create new experiment directory
ifndef $(OUTPUT_DIR)
	OUTPUT_DIR := /some/example/path/$(shell python3 ../scripts/gen.py)
endif


# Define 'all' step to run all steps
all: $(OUTPUT_DIR)/outfiles/* $(OUTPUT_DIR)/finalfiles/*


# Generate first output from provided input file
$(OUTPUT_DIR)/outfiles/*: /path/to/file/input.txt
	python3 ../scripts/outfiles.py $(OUTPUT_DIR)/outfiles/ /path/to/file/input.txt


# Generate second output from generated first output
$(OUTPUT_DIR)/finalfiles/*: $(OUTPUT_DIR)/outfiles/out.txt
	python3 ../scripts/outfiles.py $(OUTPUT_DIR)/finalfiles/ $(OUTPUT_DIR)/outfiles/out.txt
