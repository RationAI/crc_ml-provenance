from cytokeratin_mask.cytokeratin_mask_processing import get_cytokeratin_mask_holes, get_cytokeratin_mask, \
    get_hematoxylin_mask
from utils.image_tools import transform_image_by_shapely_transform


def register_hematoxylin_and_cytokeratin_mask_with_filled_holes(he, ce, transform):
    cytokeratin_mask = get_cytokeratin_mask(ce)
    hematoxylin_mask = get_hematoxylin_mask(ce)

    cytokeratin_mask_holes = get_cytokeratin_mask_holes(cytokeratin_mask, hematoxylin_mask)

    cytokeratin_mask_filled_holes = \
        (cytokeratin_mask[0:cytokeratin_mask_holes.shape[0], 0:cytokeratin_mask_holes.shape[1]] > 0) | \
        (cytokeratin_mask_holes > 0)

    ce_mask_transformed = transform_image_by_shapely_transform(cytokeratin_mask_filled_holes, transform)

    return he[0:ce_mask_transformed.shape[0], 0:ce_mask_transformed.shape[1]], ce_mask_transformed
