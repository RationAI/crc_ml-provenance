# Image registration
Performs image registration on found tissue cores of two mrxs whole slide images
and creates binary masks of epithelial areas.

## Installation:

Pull this repository.

Install all required packages and their dependencies.

### Dependencies
Can be installed using `requirements.txt`

 * histomicstk  (install using `$ python -m pip install histomicstk --find-links https://girder.github.io/large_image_wheels`)
 * memory-profiler
 * numpy
 * openslide-python
 * scikit-image
 * scipy
 * shapely
 * sklearn

Note: Does not include dependencies for package `polish_method`, which has its own dependencies.

## Usage
Command-Line Arguments \
  `--he`  - path to a .mrxs whole slide image - H&Es stain (Hematoxylin & Eosin) \
  `--ce`  - path to a .mxrs whole slide image - DAB stain (3,3'-Diaminobenzidine) \
  `--out` - output directory will create a subfolder `<slide_name>` containing `raw` and `masks` folders. Each subfolder will contain results in format `<slide_name>_<tissue_core_id>.png`

`python -m rationai.data.data.imreg [--he HE_SLIDE] [--ce DAB_SLIDE] [--out OUT_DIR] [-v --verbose]`
