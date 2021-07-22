import abc
import copy
import logging
import numpy as np
import openslide as os
import pandas as pd
from pathlib import Path
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm

from typing import Optional
from typing import NoReturn
from nptyping import NDArray

from rationai.generic import StepInterface
from rationai.utils import (
    detect_file_format,
    DirStructure,
    json_to_dict,
    Mode,
    SummaryWriter,
    ThreadSafeIterator
)

Image.MAX_IMAGE_PIXELS = None
log = logging.getLogger('visual')

# def load_visualizer(self_params, **kwargs):
#     """Loads a visualization class"""
#     class_name = self_params['class_name']

#     # Override or add keyword arguments from config file
#     # if 'config' in self_params:
#     kwargs['self_params'] = self_params.get('config')
#         # for k in self_params['config']:
#         #     kwargs[k] = identifier['config'][k]

#     path = join_module_path(__name__, class_name)
#     return load_from_module(path, **kwargs)


class VisualBase(abc.ABC):
    """Base class for WSI postprocessing"""
    def __init__(self,
                 name: str,
                 self_params: dict,
                 metadata_params: dict,
                 dir_struct: DirStructure):

        self.name = name
        self.dir_struct = dir_struct

        # Basic visualization attributes
        self.vis_level = self_params['visual_level']
        self.scale_factor = 2 ** (self.vis_level - metadata_params['level'])
        self.vis_tile_size = int(metadata_params['tile_size'] // self.scale_factor)

        if 'center_size' in metadata_params:
            self.vis_center_size = int(metadata_params['center_size'] // self.scale_factor)
            self.vis_offset = int((self.vis_tile_size - self.vis_center_size) // 2)

        # disable multiprocessing by setting to 1
        self.max_workers = int(self_params.get('max_workers', 1))

        self._create_figures_dir()

    def _create_figures_dir(self):
        """Creates folder 'figures' inside 'expdir' folder"""
        if not self.dir_struct.get('figures'):
            eval_dir = self.dir_struct.get('expdir')
            self.dir_struct.add('figures', eval_dir / 'figures', create=True)


class HeatmapVisualizer(VisualBase, StepInterface):
    """Creates probability heatmaps.

    requirements:
        Predictions have to exist in 'predictions' output directory.
    """
    def __init__(self,
                 self_params: dict,
                 metadata_params: dict,
                 dir_struct: DirStructure,
                 summary_writer: SummaryWriter):

        super().__init__(name='HeatmapVisualizer',
                         self_params=self_params,
                         metadata_params=metadata_params,
                         dir_struct=dir_struct)

        self.summary_writer = summary_writer
        # Writes results to figures/<self.name>
        self.output_dir = self._create_output_dir()
        # Reads prediction from:
        self.predict_dir = self.dir_struct.get('predictions')

    @classmethod
    def from_params(cls,
                    params: dict,
                    self_config: dict,
                    dir_struct: DirStructure,
                    summary_writer: SummaryWriter):
        """Returns initialized class or None upon failure.

        StepExecutor calls this method to instantiate the class.
        Gets entire params dict and cherry-picks needed __init__ args.
        """

        params_cp = copy.deepcopy(params)
        config = copy.deepcopy(self_config)

        try:
            return cls(
                self_params={
                    'visual_level': config.get('visual_level'),
                    'max_workers': config.get('max_workers', 1)
                },
                metadata_params=params_cp['data']['meta'],
                dir_struct=dir_struct,
                summary_writer=summary_writer)
        except Exception as e:
            log.info(f'HeatmapVisualizer from_params failed: {e}')
            return None

    def run(self, slide_pattern: str = '*'):
        """Draws probability heatmaps for slides whose predictions
        it finds in "predictions" directory.

        Args:
            slide_pattern : str
                limits computation to slides matching the pattern.
        """
        log.info(f'Running {self.name}')
        if slide_pattern != '*':
            log.info(f'Processing only predictions '
                     f'matching the pattern "{slide_pattern}"')
        slide_names = self._get_slide_names(slide_pattern)

        if not slide_names:
            log.debug("Skipping visualization. No slides found.")
            return

        with Pool(self.max_workers) as pool:
            iterator = ThreadSafeIterator(slide_names)
            # tqdm works with iterator -> imap instead of map. But imap is lazy -> result has to be used
            res = tqdm(pool.imap(self.process, iterator, chunksize=1), total=len(iterator))
            # this is the usage for lazy imap
            [_ for _ in res]
            log.info(f'Visualization done - processed heatmaps: {len(res)}')

        self.summary_writer.update_log()

    def process(self, slide_name: str) -> str:
        """Creates heatmap for a single WSI """
        if (self.output_dir / (f'{slide_name}.png')).exists():
            log.info(f'Heatmap {slide_name} exists. Skipping')
            return str(slide_name)

        log.debug(f'Processing heatmap for slide {slide_name}')
        heatmap = self._draw_heatmap_for_slide(slide_name)
        self._save_heatmap(heatmap, slide_name)
        return str(slide_name)

    def _create_output_dir(self) -> Path:
        fig_dir = self.dir_struct.get('figures')
        return self.dir_struct.add(self.name, fig_dir / self.name, create=True)

    def _load_summary(self) -> Optional[dict]:
        return json_to_dict(self.dir_struct.get('expdir'))

    def _get_slide_names(self, pattern: str = '*'):
        """Returns filenames from directory '<expdir>/preditions/'
        that match the given pattern."""
        return list(map(lambda p: p.stem, sorted(list(self.predict_dir.glob(pattern)))))

    def _draw_heatmap_for_slide(self, slide_name: str) -> NDArray:
        log.debug(f'Drawing heatmap for {slide_name}')
        slide_proba_filepath = self.predict_dir / slide_name
        slide_proba_df = pd.read_csv(slide_proba_filepath, sep=';', index_col=0)

        # Determine the size of the slide
        slide_dir = self.dir_struct.get('input')

        slide_openslide = os.open_slide(str(
            detect_file_format(folder=slide_dir,
                               pattern=f'{slide_name}.*',
                               extensions=['mrxs', 'tif', 'tiff']).resolve()))
        dimensions = slide_openslide.level_dimensions[self.vis_level]
        slide_openslide.close()

        # Prepare the canvas
        heatmap_accum_intensity = np.zeros(tuple(reversed(dimensions)), dtype='float32')
        heatmap_accum_overlap = np.zeros(tuple(reversed(dimensions)), dtype='uint8')

        # Draw the heatmap
        for _, row in slide_proba_df.iterrows():
            x = int(row.coord_x // (2**self.vis_level)) + self.vis_offset
            y = int(row.coord_y // (2**self.vis_level)) + self.vis_offset
            width = slice(x, x + self.vis_center_size)
            height = slice(y, y + self.vis_center_size)

            heatmap_accum_intensity[height, width] += row.predict
            heatmap_accum_overlap[height, width] += 1

        heatmap = np.divide(heatmap_accum_intensity, heatmap_accum_overlap,
                            out=heatmap_accum_intensity,
                            where=(heatmap_accum_overlap > 0))

        del heatmap_accum_intensity
        del heatmap_accum_overlap

        return heatmap

    def _save_heatmap(self, heatmap: NDArray, slide_name: str) -> NoReturn:
        """Saves heatmap as a numpy array"""
        log.debug(f'Saving heatmap for {slide_name}')
        heatmap_filepath = str((self.output_dir / (slide_name)).with_suffix('.png'))
        Image.fromarray((heatmap * 255).astype('uint8')).save(heatmap_filepath)

        self.summary_writer.set_value(Mode.Test.value,
                                      'summary',
                                      slide_name,
                                      'figures',
                                      'heatmap',
                                      value=str(heatmap_filepath))
        self.summary_writer.update_log()
        log.debug(f'Heatmap for {slide_name} saved')
