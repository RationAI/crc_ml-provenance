from __future__ import annotations

# Standard imports
import copy
import datetime as dt
import logging
import os
import traceback
from pathlib import Path
from typing import NoReturn
from typing import Union

# Third-party imports
import numpy as np
import openslide as oslide
import pandas as pd
from PIL import Image
from tensorflow.keras import Model

# Project-specific imports
from rationai.generic import StepInterface
from rationai.training.models import load_keras_model
from rationai.utils import DirStructure
from rationai.utils import SummaryWriter
from rationai.visual.explain import utils
from rationai.visual.explain.adapter import modeladapt
from rationai.visual.explain.compute import occlusion
from rationai.visual.explain.extern import crcmldatagen
from rationai.visual.explain.wholeslide import wsiocclusion

log = logging.getLogger('occlusion')


class OcclusionRunner(StepInterface):
    """Generates blended explanation maps,
    i.e. a single overlay for one whole slide image.

    Creates the following files in its output directory:
      - data-info.csv: metadata containing the probability for each tile at given
                       coordinates, along with the index of the tile
                       in the sequential datasource
      - result.png: the blended explanation maps converted to PNG format
    """
    def __init__(self,
                 window_size: int,
                 stride: int,
                 pixel: int,
                 interpolation: str,
                 params: dict,
                 dir_struct: DirStructure,
                 out_dir: Path = None,
                 summary_writer: SummaryWriter = None):
        """
        Args:
            window_size: int
                Occlusion window size in pixels.

            stride: int
                Occlusion window stride size in pixels.

            pixel: int
                Occlusion window color from range [0-1].

            interpolation: str
                CV2 interpolation strategy, options:
                'nearest', 'linear', 'area', 'cubic', 'lancos4'
        """

        # Occlusion computation variables
        self.window_size = window_size
        self.stride = stride
        self.pixel = pixel
        self.interpolation = interpolation

        # runner variables
        self.params = params
        self.dir_struct = dir_struct
        self.out_dir = out_dir
        self.summary_writer = summary_writer

        self.model = self._init_model()
        if summary_writer:
            self.summary_writer.set_value('occlusion', value={'mode': 'blended'})

    @classmethod
    def from_params(
            cls,
            params: dict,
            self_config: dict,
            dir_struct: DirStructure,
            summary_writer: SummaryWriter) -> Union[OcclusionRunner, None]:

        try:
            return cls(window_size=self_config.get('window_size', 10),
                       stride=self_config.get('stride', 5),
                       pixel=self_config.get('pixel', 0),
                       interpolation=self_config.get('interpolation', 'nearest'),
                       params=copy.deepcopy(params),
                       dir_struct=dir_struct,
                       out_dir=self_config.get('out_dir'),
                       summary_writer=summary_writer)
        except Exception as e:
            traceback.print_exc()
            print(f'Failed to initialize OcclusionRunner: {e}')
            return None

    def _init_model(self) -> Model:
        return modeladapt.replace_activation(
            load_keras_model(self.params['model'],
                             self.dir_struct.get('checkpoints')
                             ).test_mode().model
        )

    def run(self, slide_names: list[str]) -> NoReturn:
        """Computes occlusions for all given slides"""
        for slide_name in slide_names:
            self.process_slide(slide_name)

    def process_slide(self, slide_name: str) -> NoReturn:
        """Computes occlusions for a single WSI"""
        print(f'PROCESSING SLIDE {slide_name}')
        # DEFINE PATHS
        # NOTE: hardcoded mrxs -> add TIF
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
                                   'occlusion',
                                   slide_name,
                                   dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
        else:
            # runs differentiated by experiment id
            OUT_DIR = os.path.join(
                self.dir_struct.get('expdir'),
                'occlusion',
                slide_name
            )

        utils.make_directory_structure(OUT_DIR)

        metadata_to_save = dict(x_coord=[], y_coord=[], prob=[], image_idx=[])

        for tile_index, slide_tile in enumerate(tile_gen):
            # NOTE: first batch, first neuron
            prediction = self.model.predict(slide_tile)[0][0]

            # REMEMBER METADATA
            metadata_to_save['image_idx'].append(tile_index)
            metadata_to_save['x_coord'].append(int(tile_coords[tile_index][0]))
            metadata_to_save['y_coord'].append(int(tile_coords[tile_index][1]))
            metadata_to_save['prob'].append(prediction)

            # GET OCCLUSION MAP
            occ_handler = occlusion.OcclusionHandler(
                slide_tile[0],
                occ_size=self.window_size,
                occ_stride=self.stride,
                occ_pixel=self.pixel
            )
            occ_handler.occlude_image()
            occlusion_matrix = occ_handler.get_occlusion_matrix_no_split(
                self.model,
                batch_size=32
            )
            occlusion_matrix = prediction - occlusion_matrix
            occlusion_matrix_name = utils.get_map_name(
                tile_index, tile_coords[tile_index][0], tile_coords[tile_index][1]
            )
            np.save(
                os.path.join(OUT_DIR, occlusion_matrix_name),
                occlusion_matrix
            )

            if tile_index % 100 == 0:
                print('generated occlusion maps for', tile_index, 'slide tiles')

        # SAVE METADATA
        log.info('Computation done, saving metadata')
        df = pd.DataFrame(metadata_to_save)
        df.to_csv(os.path.join(OUT_DIR, 'data-info.csv'))

        # GET WHOLESLIDE DIMENSIONS
        slide = oslide.open_slide(str(slide_path))
        # NOTE: hardcoded level=1
        slide_dims = (
            slide.level_dimensions[1][1],
            slide.level_dimensions[1][0],
            4
        )

        # GENERATE OCCLUSION WHOLESLIDE OVERLAY
        log.info('Generating occlusion WSI overlay')
        occlusion_maps = [
            f for f in os.listdir(OUT_DIR)
            if os.path.isfile(os.path.join(OUT_DIR, f)) and
            f.endswith('.npy')
        ]
        wsiocclusion.get_occlusion_and_counts_canvas(
            slide_path,
            OUT_DIR,
            occlusion_maps,
            self.params['model'].get('input_shape')[0],
            self.params['model'].get('input_shape')[1],
            wsiocclusion.extract_coords_file_1_512_1,
            utils.get_interpolation_strategy(
                self.interpolation
            ),
            False
        )
        wsiocclusion.create_canvas(OUT_DIR, slide_path)
        log.info('Running memory map to resulting ndarray')
        canvas = np.memmap(
            os.path.join(OUT_DIR, 'result-canvas.npy'),
            dtype=np.uint8,
            mode='r',
            shape=slide_dims
        )
        log.info('Saving the result as a PNG image')
        Image.fromarray(canvas, mode='RGBA').save(
            os.path.join(OUT_DIR, 'result.png')
        )

        # CLEAN UP NOW REDUNDANT .npy FILES
        log.info('Deleting intermediate data')
        for p in Path(OUT_DIR).glob('*.npy'):
            p.unlink()
