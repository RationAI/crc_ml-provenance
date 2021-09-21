import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('prefixpath', type=Path)
parser.add_argument('inputpath', type=Path)

args = parser.parse_args()

with open(args.inputpath, 'r') as infile:
    lines = infile.readlines()
    lines.append('hello')

outdir = args.prefixpath
outdir.mkdir(parents=True, exist_ok=True)
with open(outdir / 'out.txt', 'w') as outfile:
    outfile.writelines(lines)

print('Done')

