# crcml-explainability

Explainability efforts for the CRCML cancer detection project.

## Directory structure
- `adapter`: modifications/adaptations of working objects
- `compute`: compute engines for the various explainability methods
- `config`: configuration files for the various explainability methods' CRCML processing pipelines
- `extern`: wrappers for CRCML-related code which is not maintained as part of this project
- `notebooks`: interactive Jupyter Notebooks
- `runners`: CRCML processing pipeline scripts for the various explainability methods
- `wholeslide`: joining of computed explanations into overlays for the whole slide images

## Requirements
- requirements can be installed via the `requirements.txt` file
- note that if you want to use the code from `compute.saliency`, you will also have to *manually*
  install the [Keras-vis](https://github.com/raghakot/keras-vis) package (recommended to use
  latest master commit)