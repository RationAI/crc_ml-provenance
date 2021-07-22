import argparse
import numpy as np
import re

import h5py
from pathlib import Path
from time import time
from typing import Dict
from typing import Union

from rationai.utils import mkdir

Numeric = Union[int, float]


def get_metadata(ds_filename: str) -> Dict[str, Numeric]:
    def decode_min_max(s: str) -> float:
        """Converts string represetation to float.
        e.g.: "1" -> 1.0; "075" -> 0.75
        """
        if s == '1':
            return 1.0
        alist = list(s)
        if '.' not in alist:
            alist.insert(1, '.')
        return float(''.join(alist))

    match = re.search(
        '(?P<ds_prefix>.*)-'
        'L(?P<level>.*)-'
        'T(?P<tile_size>.*)-'
        'S(?P<step_size>.*)-'
        'C(?P<center_size>.*)-'
        'MIN(?P<min_tissue>.*)-'
        'MAX(?P<max_tissue>[0-9][0-9]*)',
        ds_filename
    )

    if match is None:
        return None

    res = match.groupdict()
    res['timestamp'] = int(time())
    res['min_tissue'] = decode_min_max(res['min_tissue'])
    res['max_tissue'] = decode_min_max(res['max_tissue'])

    # Convert numeric strings to int
    for k, v in res.items():
        res[k] = int(v) if type(v) is str and v.isnumeric() else v
    return res


def main(args):

    base_dir = args.base_dir

    if not base_dir.exists():
        raise FileNotFoundError(f'Directory "{base_dir}" does not exist.')

    # Get all coordinate maps in base dir
    train_maps = np.array([p.relative_to(base_dir)
                           for p in args.coord_maps.glob(args.train)],
                          dtype=object)
    test_maps = np.array([p.relative_to(base_dir)
                          for p in args.coord_maps.glob(args.test)],
                         dtype=object)

    if len(train_maps) == 0:
        raise ValueError('Found no training maps.')
    else:
        print(f'Found {len(train_maps)} training maps.')
    if len(test_maps) == 0:
        raise ValueError('Found no test maps.')
    else:
        print(f'Found {len(test_maps)} testing maps.')

    # Output dir
    output_dir = base_dir / 'datasets'
    mkdir(output_dir)

    # Create HDF5 dataset
    ds_filename = f'{args.ds_prefix}-{args.coord_maps.name}' \
        if args.ds_prefix else args.coord_maps.name

    # Add timestamp if such dataset already exists
    if (output_dir / (f'{ds_filename}.h5')).exists():
        ds_filename += '-' + str(round(time()))

    with h5py.File((output_dir / ds_filename).with_suffix('.h5'), 'w') as f:
        string_dt = h5py.special_dtype(vlen=str)
        f.create_dataset('train', data=train_maps, dtype=string_dt)
        f.create_dataset('test', data=test_maps, dtype=string_dt)
        meta = get_metadata(ds_filename)
        if meta:
            f['train'].attrs.update(meta)
        # Creates HDF5 Attribute containing classes stratified split
        # NOTE: condition is data specific (Path.stem[-1])
        f['train'].attrs.create('stratify', list(map(lambda p: p.stem[-1], train_maps)))
        print(f'Dataset created: {output_dir / ds_filename}.h5')


if __name__ == '__main__':
    description = """
    Creates an HDF5 file that can used by the pipeline as a dataset

    Example usage:
    --------------
    $ python -m rationai.data.classify.create_dataset
                -b data/Prostata
                -c coord_maps/L1-T512-S128-C256-MIN05-MAX1  # relative to base_dir or absolute
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-b', '--base_dir', required=True, type=Path, help='Data base dir')
    parser.add_argument('-c', '--coord_maps', required=True, type=Path, help='Folder with coord_maps. Use relative path if base_dir belongs parents of coord_maps.')
    parser.add_argument('-p', '--ds_prefix', required=False, type=str, default='', help='Dataset name prefix to differentiate files')
    parser.add_argument('--train', required=False, type=str, default='P*.gz', help='Pattern for training slides')
    parser.add_argument('--test', required=False, type=str, default='TP*.gz', help='Pattern for test slides')

    args = parser.parse_args()

    args.coord_maps = args.base_dir / args.coord_maps

    main(args)
