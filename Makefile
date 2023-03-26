# Variables to set when running the script
# TRAIN_CONFIG, TEST_CONFIG, EVAL_CONFIG, EID_PREFIX

PIP=pip3
PYTHON=python3
EID := $(shell openssl rand -hex 12)

# Prepends EID Prefix if defined
ifdef EID_PREFIX
EID := $(EID_PREFIX)-$(EID)
endif

.PHONY: setup convert train test meta

run: train test meta resolve

setup: requirements.txt
	$(PIP) install -r requirements.txt
    
convert: $(CONVERT_CONFIG)
	$(PYTHON) -m rationai.data.tiler.xml_annot_patcher --config_fp $(CONVERT_CONFIG)
	$(PYTHON) -m rationai.provenance.data.tiler.xml_annot_patcher --config_fp $(CONVERT_CONFIG)

train: $(TRAIN_CONFIG)
	$(PYTHON) -m rationai.training.experiments.slide_train --config_fp $(TRAIN_CONFIG) --eid $(EID)
	$(PYTHON) -m rationai.provenance.experiments.slide_train --config_fp $(TRAIN_CONFIG) --eid $(EID)

test: $(TEST_CONFIG)
	$(PYTHON) -m rationai.training.experiments.slide_test --config_fp $(TEST_CONFIG) --eid $(EID)
	$(PYTHON) -m rationai.provenance.experiments.slide_test --config_fp $(TEST_CONFIG) --eid $(EID)
    
meta:
	$(PYTHON) -m rationai.provenance.meta.meta_prov --config_fp $(TEST_CONFIG) --eid $(EID)
    
resolve:
	$(PYTHON) -m rationai.provenance.export --config_fp $(TEST_CONFIG) --eid $(EID)
    
rocrate:
	$(PYTHON) -m rationai.provenance.rocrate.ro_create --rocrate_log $(ROCRATE_LOG)
    
provenance:
	$(PYTHON) -m rationai.provenance.data.tiler.xml_annot_patcher --config_fp $(CONVERT_CONFIG)
	$(PYTHON) -m rationai.provenance.experiments.slide_train --config_fp $(TRAIN_CONFIG) --eid $(EID)
	$(PYTHON) -m rationai.provenance.experiments.slide_test --config_fp $(TEST_CONFIG) --eid $(EID)
	$(PYTHON) -m rationai.provenance.meta.meta_prov --config_fp $(TEST_CONFIG) --eid $(EID)
	$(PYTHON) -m rationai.provenance.export --config_fp $(TEST_CONFIG) --eid $(EID)
