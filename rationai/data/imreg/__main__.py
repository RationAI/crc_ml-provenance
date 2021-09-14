import argparse

from rationai.data.imreg.main import ImageRegistration

parser = argparse.ArgumentParser()

# Required args
parser.add_argument('--he',  type=str, required=True, help='HE mrxs file')
parser.add_argument('--ce',  type=str, required=True, help='HDAB mrxs file')
parser.add_argument('--out', type=str, required=True, help='output directory')

# Optional args
parser.add_argument('--verbose', '-v', action='store_true', help='Turns on verbosity')

args = parser.parse_args()

img_reg = ImageRegistration(he_mrxs_path   = args.he,
                            hdab_mrxs_path = args.ce,
                            output_dir     = args.out,
                            verbose        = args.verbose)

img_reg.run()
