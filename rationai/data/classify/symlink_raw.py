import argparse
from pathlib import Path

SLIDE_SUB_DIR = 'slides'  # symlinks to .mrxs files
ANNOT_SUB_DIR = 'label'  # symlinks to .xml files


def symlink_raw(input_dir, output_dir, max_count):
    """Creates a symlink for every .mrxs slide and its annotation (if exists)
        from source directory to target directory.

        Two new directories will be created in the target directory
        <SLIDE_SUB_DIR> and <ANNOT_SUB_DIR> into which symlinks
        will be created.

    Arguments:
        source {Path} -- path to source directory
        target {Path} -- path to target directory
        max {int}     -- max number of slides to process (optional arg)
    """

    # Create output slide directory
    out_slide_dir = output_dir / SLIDE_SUB_DIR
    if not out_slide_dir.exists():
        out_slide_dir.mkdir(parents=True)

    # Create output annotation directory
    out_annot_dir = output_dir / ANNOT_SUB_DIR
    if not out_annot_dir.exists():
        out_annot_dir.mkdir(parents=True)

    for ext in ['mrxs', 'tif']:
        counter = 0
        for slide in input_dir.glob(f'**/*.{ext}'):
            # Create symlink if does not exists
            target_slide_fn = (out_slide_dir / slide.name)
            if not target_slide_fn.exists():
                target_slide_fn.symlink_to(slide)

            # Infer annotation and symlink if does not exists
            src_xml_fn = slide.with_suffix('.xml')
            target_xml_fn = (out_annot_dir / src_xml_fn.name)
            if src_xml_fn.exists() and not target_xml_fn.exists():
                target_xml_fn.symlink_to(src_xml_fn)

            counter += 1
            if max_count is not None and counter == max_count:
                break
        print(f'Found {counter} slides in {ext.upper()} format.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Creates symlink for every .mrxs slide and its .xml annotation.')

    parser.add_argument('-s', '--source', dest='src', metavar='SRC', type=Path, required=True,
                        help='Source directory. Will be searched recursively for all .mrxs slides.')
    parser.add_argument('-t', '--target', dest='tgt', metavar='TGT', type=Path, required=True,
                        help="Target directory. Will create (if not exist already) "
                             "two new directories 'slides' and 'annotations'.")

    # Max number of slides can be useful for small debug datasets
    parser.add_argument('-m', '--max', type=int, default=None,
                        help='Max number of created symlinks.')

    args = parser.parse_args()
    symlink_raw(args.src, args.tgt, args.max)
