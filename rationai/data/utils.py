import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from PIL import UnidentifiedImageError

from typing import (
    List,
    Optional,
    Tuple,
    Union
)


Vertices = List[Tuple[float, float]]

def open_pil_image(path: Union[str, Path]) -> Optional[Image.Image]:
    """Opens and returns an Image.
    Returns None if Image is corrupted or does not exist."""
    try:
        return Image.open(str(path))
    except (UnidentifiedImageError, FileNotFoundError):
        return None


def read_polygons(annotation_filepath: Path,
                  scale_factor: float,
                  keywords: List[str]) -> Tuple[Vertices, Vertices]:
    """Utility function to read an annotation XML file and create
    a list of vertices for polygon delimiting the cancerous area.
    """
    if not annotation_filepath.exists():
        return [], []

    # Read cancer polygon area
    polygons = []

    root = ET.parse(str(annotation_filepath)).getroot()
    for anno_tag in root.findall('Annotations/Annotation'):
        polygon = []

        keyword = anno_tag.get('PartOfGroup')

        if keyword in keywords:
            for coord in anno_tag.findall('Coordinates/Coordinate'):
                polygon.append((float(coord.get('X')) / scale_factor,
                                float(coord.get('Y')) / scale_factor))

        polygons.append(polygon)

    return polygons
