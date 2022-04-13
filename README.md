# Histopat Provenanace

This repository-branch implements an example of a distributed provenance model generation. It is applied on a histopathological ML pipeline primarily used for cancer detection research.

The branch contains a portion of the Histopat pipeline necessary to run an example. 

## ML Pipeline

The pipeline works with units called Experiments. An Experiment defines a logic of a job to be run using a configuration file. A configuration file is a nested JSON file describing the following:

- **Definitions** - defines what components (Data, Generator, Model, Callbacks, etc) are to be used in the experiment
- **Configurations** - defines the parameters of the components

Sample configuration files can be found in `rationai/config/` directory. The pipeline can be run using the provided Makefile files:

**Slide conversion**
`make -f Makefile.convert run CONFIG_FILE=rationai/config/prov_converter_config.json`

**Experiments**
`make -f Makefile.experiment run TRAIN_CONFIG=rationai/config/prov_train_config.json TEST_CONFIG=rationai/config/prov_test_config.json EVAL_CONFIG=rationai/config/prov_eval_config.json EID_PREFIX=PROV`

alternatively, each experiment can be run individually

`make -f Makefile.experiment setup train TRAIN_CONFIG=rationai/config/prov_train_config.json EID_PREFIX=PROV-TRAIN` 

Each makefile call creates a new experiment directory `<EID_PREFIX>-<EID_HASH>` where `EID_PREFIX` can be set during the Makefile call for easier experiment identification and `EID_HASH` is generated randomly to minimze experiment overwriting.

## Provenance Logging

During a regular run of an experiment a structured JSON log is being constructed using a custom `SummaryWriter` object. Only a single copy with a given name can exist at any given time. Retrieveing a `SummaryWriter` object with the same name from multiple locations results in the same object similarly to standard `logging.Logger`. 

Any key-value pair that we wish to keep track of must be set using the `SummaryWriter` `.set()` or `.add()` functions. The utility package `rationai.utils.provenance` contains additional helpful functions for generating provenanace such as SHA256 hashing function for pandas tables, pandas HDFStore, filepaths and directories.

```
log = SummaryWriter.getLogger('provenance')
log.set('level1', 'level2, 'level3', value='value')
log.set('level1', 'key', value=5)
log.to_json(filepath)

# {
#     'level1': {
#         'level2': {
#             'level3': 'value'
#         },
#         'key': 5
#     }
# }

```

## Provenanace generation

In order to parse the logs and generate provenanace graph we can call the `Makefile.provenance` file.

**Provenance Graph Generation**
`make -f Makefile.provenance run TRAIN_LOG=experiments/8c85b9321e00eeac082da2c3/prov_train.log TEST_LOG=experiments/8c85b9321e00eeac082da2c3/prov_test.log EVAL_LOG=experiments/8c85b9321e00eeac082da2c3/prov_eval.log PREPROC_LOG=data/prov_preprocess.log`


