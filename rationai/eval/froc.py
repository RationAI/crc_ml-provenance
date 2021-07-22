import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing import Pool
from skimage.io import imread
from shapely.geometry.polygon import Polygon
from sklearn.metrics import auc
from scipy.interpolate import interp1d

from typing import Tuple
from nptyping import NDArray

from rationai.data.utils import read_polygons
from rationai.generic import StepInterface
from rationai.utils import (
    DirStructure,
    Mode,
    SummaryWriter,
    ThreadSafeIterator
)
from rationai.visual import VisualBase

Image.MAX_IMAGE_PIXELS = None
log = logging.getLogger('eval')


class FROC(VisualBase, StepInterface):
    """FROC score computation.

    The class requires predictions and probability heatmaps to exist.
    """
    def __init__(self,
                 self_params: dict,
                 metadata_params: dict,
                 dir_struct: DirStructure,
                 summary_writer: SummaryWriter):

        super().__init__(name="FROC",
                         self_params=self_params,
                         metadata_params=metadata_params,
                         dir_struct=dir_struct)

        self.summary_writer = summary_writer
        self.verbose = self_params.get('verbose', 0)

        # NMS computation parameters
        self.nms_threshold = self_params.get('nms_threshold', 0.5)
        self.nms_radius = self_params.get('radius', 1)
        self.neighborhood = self_params.get('neighborhood', 'google')

        # NOTE: think about a way how to un hard code this.
        self.heatmap_dir = dir_struct.get('HeatmapVisualizer')

        # WSI annotations designating cancerous areas
        self.include_polygons = metadata_params.get('include_annot_keywords', [])
        # WSI annotations designating regions to ignore
        self.exclude_polygons = metadata_params.get('exclude_annot_keywords', [])

        if self.heatmap_dir is None:
            raise ValueError('Path to heatmaps not found in path manager.')
        if len(self.include_polygons) == 0:
            raise ValueError('Metadata "include_annot_keywords" not found or empty.')

    @classmethod
    def from_params(cls,
                    params: dict,
                    self_config: dict,
                    dir_struct: DirStructure,
                    summary_writer: SummaryWriter):
        """Returns instantiated class or None upon failure.
        StepExecutor calls this method to initialize the class.
        Gets entire params dict and cherry-picks needed __init__ args.
        """
        params_cp = copy.deepcopy(params)
        config = copy.deepcopy(self_config)

        try:
            return cls(
                self_params={
                    'visual_level': config.get('visual_level'),
                    'nms_threshold': config.get('nms_threshold', 0.5),
                    'nms_radius': config.get('nms_radius', 1),
                    'nms_neighbourhood': config.get('nms_neighbourhood', 'google'),
                    'max_workers': config.get('max_workers', 1),
                    'verbose': config.get('verbose', 0)
                },
                metadata_params=params_cp['data']['meta'],
                dir_struct=dir_struct,
                summary_writer=summary_writer)
        except Exception as e:
            log.info(f'FROC from_params failed: {e}')
            return None

    def run(self, heatmap_pattern: str = '*') -> float:
        """Computes and returns FROC score.

        Keyword arguments:
        heatmap_pattern -- limits the computation to the specific slides
        """
        log.info('Computing FROC')

        # Non Maximum Suppresion computation
        all_dfs_nms = self._distribute_nms_computation(heatmap_pattern)

        FROC_THRESHOLDS = np.arange(1.0, 0.50, -0.0005)
        sens_l = []  # sensitivity
        avfp_l = []  # average False Positives

        for froc_thresh in FROC_THRESHOLDS:
            # FROC-level counters
            total_counter = {'TP': 0, 'FP': 0, 'tumours': 0, 'neg_slides': 0}

            # For each slide, compute TPs or FPs depending on the slide label
            for df_nms, num_polygons in np.array(all_dfs_nms):
                # slide-level counters
                slide_counter = {'TP': 0, 'FP': 0}

                if len(df_nms) > 0:
                    if num_polygons > 0:
                        # TPs are measured only in positive slides
                        slide_counter['TP'] = np.sum(df_nms[df_nms.conf >= froc_thresh]
                                                     .polygon.unique() >= 0)
                    else:
                        # FPs are measured only in negative slides
                        slide_counter['FP'] = df_nms[df_nms.conf >= froc_thresh] \
                            .polygon.lt(0).sum()
                        total_counter['neg_slides'] += 1

                # Accumulate metrics
                total_counter['TP'] += slide_counter['TP']
                total_counter['FP'] += slide_counter['FP']
                total_counter['tumours'] += num_polygons

            # Calculate FROC statistics for given threshold
            sens_l.append(total_counter['TP'] / (
                total_counter['tumours'] + np.finfo(np.float32).eps))
            avfp_l.append(total_counter['FP'] / (
                total_counter['neg_slides'] + np.finfo(np.float32).eps))

        # Only keep those below 8 AVG FP
        f = np.array(avfp_l) <= 8

        x, y = np.round(np.array(avfp_l)[f], 6), np.round(np.array(sens_l)[f], 6)

        # Calculate & save results
        if len(x) < 2:
            self.froc_score = 0.0
            log.info(
                'Predictions contain too many False Positives per WSI on average. '
                '(Cannot compute AUC with less than 2 values)')
        else:
            self.froc_score = auc(x=x, y=y) / 8

        self._save_results(x, y)
        self._generate_plot(x, y)
        self.summary_writer.update_log()
        return self.froc_score

    def _distribute_nms_computation(self, heatmap_pattern: str = '*') -> list:
        log.info('Distributing NMS computation among workers')

        if heatmap_pattern != '*':
            log.info('Processing only slides '
                     f'matching the pattern "{heatmap_pattern}"')

        slide_paths = list(sorted(self.heatmap_dir.glob(heatmap_pattern)))
        if not slide_paths:
            log.info(f'No PNG heatmaps found in dir {self.heatmap_dir}')
            return []

        with Pool(self.max_workers) as pool:
            iterator = ThreadSafeIterator(slide_paths)
            res = tqdm(pool.imap(self._process_path, iterator), total=len(iterator))
            return [x for x in res]

    def _process_path(self, slide_path: Path) -> Tuple[pd.DataFrame, int]:
        """Initiates NMS computation for a single slide"""
        slide_name = slide_path.stem
        log.debug(f'Computing NMS for {slide_name}')

        # load heatmap & df
        heatmap = imread(self.heatmap_dir / (slide_name + '.png'))
        df_result = pd.read_csv(self.dir_struct.get('predictions') / slide_name, sep=';')

        # compute nms
        annot_path = self.dir_struct.get('label') / (slide_name + '.xml')

        include_polygons, _ = read_polygons(
            annot_path,
            2**self.vis_level,
            include_keywords=self.include_polygons,
            exclude_keywords=self.exclude_polygons)

        nms_points, num_polygons = self._nms_from_dataframe(df_result.copy(),
                                                            heatmap,
                                                            include_polygons)

        if len(nms_points) > 0:
            idx, conf, polygon_id = zip(*nms_points)
        else:
            idx, conf, polygon_id = [], [], []

        df_nms = pd.DataFrame({'idx': idx, 'conf': conf, 'polygon': polygon_id})

        log.debug(f'NMS computation for {slide_name} finished')
        return df_nms, num_polygons

    def _nms_from_dataframe(self,
                            df_result: pd.DataFrame,
                            heatmap: NDArray,
                            include_polygons: list):
        """Computes NMS for given results and a heatmap"""
        # Helper structures to count metrics
        nms_polygons = [Polygon(polygon) for polygon in include_polygons]
        nms_found = np.zeros_like(nms_polygons, dtype='bool')

        # NMS Points (result)
        nms_points = []

        # Extract max tile
        max_idx = df_result['predict'].idxmax()
        max_row = df_result.iloc[max_idx]

        # Verify above threshold
        is_valid = max_row.predict >= self.nms_threshold

        while is_valid:
            # Extract coordinates
            coord_x = int(max_row.coord_x // 2 ** self.vis_level)
            coord_y = int(max_row.coord_y // 2 ** self.vis_level)

            # Verify it was not suppressed
            is_suppressed = heatmap[coord_y, coord_x] == 0
            if self.verbose > 1:
                log.debug(f'Suppressed: {is_suppressed}')

            if not is_suppressed:

                if self.neighborhood == 'context':
                    # Context radius
                    start_x = (coord_x + self.vis_offset) - (self.nms_radius * self.vis_center_size)
                    end_x = (coord_x + self.vis_offset) + ((self.nms_radius + 1) * self.vis_center_size)
                    start_y = (coord_y + self.vis_offset) - (self.nms_radius * self.vis_center_size)
                    end_y = (coord_y + self.vis_offset) + ((self.nms_radius + 1) * self.vis_center_size)
                elif self.neighborhood == 'google':
                    # Possible Google context
                    start_x = (coord_x + self.vis_offset) - (self.nms_radius * self.vis_center_size)
                    end_x = (coord_x + self.vis_offset) + (self.nms_radius * self.vis_center_size) + 1
                    start_y = (coord_y + self.vis_offset) - (self.nms_radius * self.vis_center_size)
                    end_y = (coord_y + self.vis_offset) + (self.nms_radius * self.vis_center_size) + 1
                else:
                    raise ValueError('Unknown neighborhood type.')

                heatmap[start_y:end_y, start_x:end_x] = 0

                test_polygon = Polygon([(start_x, start_y), (end_x, start_y), (end_x, end_y), (start_x, end_y)])
                nms_intersects = np.array([nms_polygon.intersects(test_polygon) for nms_polygon in nms_polygons])

                if np.any(nms_intersects):
                    for polygon_id in np.nonzero(nms_intersects)[0]:
                        nms_points.append((max_idx, max_row.predict, polygon_id))
                    nms_found[np.nonzero(nms_intersects)] = True
                else:
                    nms_points.append((max_idx, max_row.predict, -1))

            df_result.at[max_idx, 'predict'] = 0.0

            # Extract max tile
            max_idx = df_result['predict'].idxmax()
            max_row = df_result.iloc[max_idx]

            # Verify above threshold
            is_valid = max_row.predict >= self.nms_threshold

        return nms_points, len(nms_polygons)

    def _generate_plot(self, avfp_ndarray: NDArray, sens_ndarray: NDArray):
        """Create and save FROC curve plot"""
        log.debug('Generating FROC curve plot')
        label = f'FROC curve (area = {self.froc_score:.4})'
        plt.figure(dpi=180)

        ax = plt.gca()
        ax.get_yaxis().set_tick_params(direction='in', right=True)
        ax.get_xaxis().set_tick_params(direction='in', top=True)

        plt.plot(avfp_ndarray, sens_ndarray, color='blue', lw=1, label=label)

        plt.title('FROC curve')
        plt.xlim([0, 8])
        plt.ylim([0.0, 1.0])
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        plt.xlabel('Average Number of False Positives')
        plt.ylabel('Metastasis detection sensitivity')
        plt.legend(loc="lower right",
                   fontsize='large',
                   fancybox=False,
                   framealpha=1,
                   edgecolor='black')
        plt.grid(True, linestyle='--', color='black', lw=0.5, dashes=(2, 6, 2, 6))

        plt.savefig(self.heatmap_dir.parent / 'froc-curve.png')

    def _save_results(self, avfp_ndarray: NDArray, sens_ndarray: NDArray):
        """Save results to summary.json"""

        # FROC score
        log.info(f'FROC score = {self.froc_score:.4f}')
        self.summary_writer.set_value(Mode.Eval.value, 'froc', value=self.froc_score)

        if self.froc_score == 0:
            return

        # Average specificity for each predefined point
        inter_f = interp1d(x=avfp_ndarray, y=sens_ndarray) \
            if len(avfp_ndarray) >= 2 else lambda x: 0

        for k, x in zip(['1/4FP', '1/2FP', '1FP', '2FP', '4FP', '8FP'],
                        [0.25, 0.5, 1, 2, 4, 8]):
            try:
                avg_sp = float(inter_f(x))
            except ValueError as e:
                log.info(f'scipy.interp1d interpolation error: {e}')
                avg_sp = np.nan

            log.info(f'Avg specificity at {k} = {avg_sp:.4f}')
            self.summary_writer.set_value(
                Mode.Eval.value, 'avg_sensitivity_at', k, value=avg_sp)
