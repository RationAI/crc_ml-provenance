from __future__ import annotations

# Standard imports
import copy
import datetime as dt
import logging
import os
from pathlib import Path
from typing import (
    List,
    NoReturn,
    Union
)

# Third-party imports
import numpy as np
import openslide as oslide
import pandas as pd
from PIL import Image
from tensorflow.keras import Model

# Project-specific imports
from rationai.generic import StepInterface
from rationai.utils import DirStructure
from rationai.utils import SummaryWriter
from rationai.training.models import load_keras_model
from rationai.visual.explain import utils
from rationai.visual.explain.adapter import modeladapt
from rationai.visual.explain.compute import saliencytf2
from rationai.visual.explain.extern import crcmldatagen
from rationai.visual.explain.wholeslide import wsisaliency

log = logging.getLogger('saliency')


class SaliencyRunner(StepInterface):
    """Generates blended explanation maps,
    i.e. a single overlay for one whole slide image.

    Creates the following files in its output directory:
      - data-info.csv: metadata containing the probability for each tile
                       at given coordinates, along with the index of the tile
                       in the sequential datasource
      - result-[neg/pos].png: the blended explanation maps converted to PNG format
    """
    def __init__(self,
                 grad_modifier: str,
                 params: dict,
                 dir_struct: DirStructure,
                 out_dir: Path = None,  # just leave it as None
                 summary_writer: SummaryWriter = None):
        """
        Args:
            grad_modifier: str
                Gradient modifier functions, options: [None, 'absolute', 'relu']

        """

        self.grad_modifier = grad_modifier
        self.params = params
        self.dir_struct = dir_struct
        self.out_dir = out_dir
        self.summary_writer = summary_writer

        self.model = self._init_model()
        if summary_writer:
            self.summary_writer.set_value('saliency', value={'mode': 'blended'})

    @classmethod
    def from_params(
            cls,
            params: dict,
            self_config: dict,
            dir_struct: DirStructure,
            summary_writer: SummaryWriter) -> Union[SaliencyRunner, None]:

        try:
            return cls(grad_modifier=self_config.get('grad_modifier'),
                       params=copy.deepcopy(params),
                       dir_struct=dir_struct,
                       out_dir=self_config.get('out_dir'),
                       summary_writer=summary_writer
                       )
        except Exception as e:
            print(f'Failed to initialize SaliencyRunner: {e}')
            return None

    def _init_model(self) -> Model:
        return modeladapt.replace_activation(
            load_keras_model(self.params['model'],
                             self.dir_struct.get('checkpoints')
                             ).test_mode().model
        )

    def run(self, slide_names: List[str]) -> NoReturn:
        """Computes saliency maps for all given slides

        Args:
            slide_names : List[str]
                List of names of slides to process.
        """

        for slide_name in slide_names:
            self.process_slide(slide_name)

    def process_slide(self, slide_name: str) -> NoReturn:
        """Computes saliency map for a single WSI"""
        print(f'PROCESSING SLIDE {slide_name}')
        # DEFINE PATHS
        slide_path = (self.dir_struct.get('input') / f'{slide_name}.mrxs').resolve()
        coord_map_path = self.dir_struct.get('coord_maps') / f'{slide_name}.gz'

        # TILE COORD
        tile_coords = utils.extract_coordinates(coord_map_path)

        # GENERATOR
        tile_gen = crcmldatagen.get_datasource_for_test_slide_tf2(
            params=copy.deepcopy(self.params),
            coord_map_path=coord_map_path,
            dir_struct=self.dir_struct)

        # DIRS
        if self.out_dir:
            # runs differentiated by datetime
            OUT_DIR = os.path.join(self.out_dir,
                                   'saliency',
                                   slide_name,
                                   dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
        else:
            # runs differentiated by experiment id
            OUT_DIR = os.path.join(
                self.dir_struct.get('expdir'),
                'saliency',
                slide_name
            )

        utils.make_directory_structure(OUT_DIR)

        metadata_to_save = dict(x_coord=[], y_coord=[], prob=[], image_idx=[])
        visualizer = saliencytf2.SaliencyVis(self.model)
        grad_modifier = saliencytf2.get_grad_modifier(self.grad_modifier)

        for tile_index, slide_tile in enumerate(tile_gen):
            prediction = self.model.predict(slide_tile)[0][0]

            # REMEMBER METADATA
            metadata_to_save['image_idx'].append(tile_index)
            metadata_to_save['x_coord'].append(int(tile_coords[tile_index][0]))
            metadata_to_save['y_coord'].append(int(tile_coords[tile_index][1]))
            metadata_to_save['prob'].append(prediction)

            # GET SALIENCY MAP FOR ALL COLOR CHANNELS
            sal_map = visualizer.visualize_saliency(
                slide_tile,
                saliencytf2.sal_loss,
                keep_dims=True,
                normalize=False,
                gradient_modifier=grad_modifier
            )
            saliency_map_name = utils.get_map_name(
                tile_index,
                tile_coords[tile_index][0],
                tile_coords[tile_index][1]
            )

            np.save(os.path.join(OUT_DIR, saliency_map_name), sal_map)

            # GET MAXIMUM POSITIVE VALUE SALIENCY MAP
            positive_map = np.zeros(sal_map.shape)
            positive_map[np.where(sal_map > 0)] = sal_map[np.where(sal_map > 0)]
            positive_map = np.max(positive_map, axis=-1)
            positive_map_name = utils.get_map_name(
                tile_index,
                tile_coords[tile_index][0],
                tile_coords[tile_index][1],
                '-pos.npy'
            )

            np.save(os.path.join(OUT_DIR, positive_map_name), positive_map)

            # GET MINIMUM NEGATIVE VALUE SALIENCY MAP
            negative_map = np.zeros(sal_map.shape)
            negative_map[np.where(sal_map < 0)] = sal_map[np.where(sal_map < 0)]
            negative_map = np.min(negative_map, axis=-1)
            negative_map_name = utils.get_map_name(
                tile_index,
                tile_coords[tile_index][0],
                tile_coords[tile_index][1],
                '-neg.npy'
            )
            np.save(os.path.join(OUT_DIR, negative_map_name), negative_map)

            if tile_index % 100 == 0:
                print('generated saliency maps for', tile_index, 'slide tiles')

        # SAVE METADATA
        log.info('Computation done, saving metadata')
        df = pd.DataFrame(metadata_to_save)
        df.to_csv(os.path.join(OUT_DIR, 'data-info.csv'))

        # GET WHOLESLIDE DIMENSIONS
        slide = oslide.open_slide(str(slide_path))
        slide_dims = (
            slide.level_dimensions[1][1],
            slide.level_dimensions[1][0],
            4
        )

        # GENERATE POSITIVE SALIENCY WHOLESLIDE OVERLAY
        log.info('Generating positive saliency WSI overlay')
        positive_maps = [
            f for f in os.listdir(OUT_DIR)
            if os.path.isfile(os.path.join(OUT_DIR, f))
            and f.endswith('-pos.npy')
        ]
        wsisaliency.get_saliency_and_counts_canvas(
            slide_path,
            OUT_DIR,
            positive_maps,
            wsisaliency.extract_coords_file_1_512_1,
            pos=True,
            border=False
        )
        wsisaliency.create_canvas(
            OUT_DIR, slide_path, pos=True
        )
        canvas = np.memmap(
            os.path.join(OUT_DIR, 'result-canvas-pos.npy'),
            dtype=np.uint8,
            mode='r',
            shape=slide_dims
        )
        log.info('Saving the result as a PNG image')
        Image.fromarray(canvas, mode='RGBA').save(
            os.path.join(OUT_DIR, 'result-pos.png')
        )

        # GENERATE NEGATIVE SALIENCY WHOLESLIDE OVERLAY
        log.info('Generating negative saliency WSI overlay')
        negative_maps = [
            f for f in os.listdir(OUT_DIR)
            if os.path.isfile(os.path.join(OUT_DIR, f))
            and f.endswith('-neg.npy')
        ]
        wsisaliency.get_saliency_and_counts_canvas(
            slide_path,
            OUT_DIR,
            negative_maps,
            wsisaliency.extract_coords_file_1_512_1,
            pos=False,
            border=False
        )
        wsisaliency.create_canvas(
            OUT_DIR, slide_path, pos=False
        )
        canvas = np.memmap(
            os.path.join(OUT_DIR, 'result-canvas-neg.npy'),
            dtype=np.uint8,
            mode='r',
            shape=slide_dims
        )
        log.info('Saving the result as a PNG image')
        Image.fromarray(canvas, mode='RGBA').save(
            os.path.join(OUT_DIR, 'result-neg.png')
        )

        # CLEAN UP NOW REDUNDANT .npy FILES
        log.info('Deleting intermediate data')
        for p in Path(OUT_DIR).glob('*.npy'):
            p.unlink()
