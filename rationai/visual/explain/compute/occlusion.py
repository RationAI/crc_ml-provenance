"""
Occlusion computation engine.
"""
# Standard imports
import math

# Third-party imports
import numpy as np


class OcclusionHandler:
    """
    Handler for occlusion of a single image.

    Usage:
    handler = OcclusionHandler(...)
    handler.occlude_image()
    occ_matrix = handler.get_occlusion_matrix(...)

    Attributes
    ----------
    image : array-like
        Number array representing the image to occlude.
    image_height : int
        Height of `image`.
    img_width : int
        Width of `image`.
    occ_size : int
        Size of the occlusion window in pixels (length of side of occlusion
        square). Default 10.
    occ_stride : int
        Size of the stride of the occlusion window in pixels. Default 5.
    occ_pixel : int
        Color of the occlusion window (range from 0 to 1). Default 0.
    output_height : int
        Height of the output occlusion matrix.
    output_width : int
        Width of the output occlusion matrix.
    occluded_subimages : list(array-like)
        List of the occluded images, based on the init parameters. Available
        after calling `occlude_image`.
    """

    def __init__(self, image, occ_size=10, occ_stride=5, occ_pixel=0):
        self.image = image
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]
        self.occ_size = occ_size
        self.occ_stride = occ_stride
        self.occ_pixel = occ_pixel
        self.output_height = int(
            math.ceil(
                (self.image_height - self.occ_size) / self.occ_stride + 1
            )
        )
        self.output_width = int(
            math.ceil(
                (self.image_width - self.occ_size) / self.occ_stride + 1
            )
        )
        self.occluded_subimages = list()

    def occlude_image(self):
        """
        Generates occluded images of input image given at initialization.

        The list of occluded images is stored in the `occluded_subimages`
        attribute.
        """
        for h in range(self.output_height):
            for w in range(self.output_width):
                self.occluded_subimages.append(
                    (self.get_occluded_subimage(h, w), h, w)
                )

    def get_occluded_subimage(self, occ_coord_x, occ_coord_y):
        """
        Creates an occluded image.

        Parameters
        ----------
        occ_coord_x : int
            Top coordinate of the occlusion square.
        occ_coord_y : int
            Leftmost coordinate of the occlusion square.

        Returns
        -------
            Original image, with the square given by `occ_coord_x`,
            `occ_coord_y` and the `occ_size` attribute.
        """
        h_start = occ_coord_x * self.occ_stride
        w_start = occ_coord_y * self.occ_stride
        h_end = min(self.image_height, h_start + self.occ_size)
        w_end = min(self.image_width, w_start + self.occ_size)
        occ_image = self.image.copy()
        occ_image[h_start:h_end, w_start:w_end] = self.occ_pixel
        return occ_image

    def get_occlusion_matrix_no_split(self, model, batch_size=64):
        """
        Compute the occlusion matrix based on `model` and handler attributes.

        Should be called only after `occlude_image` has been called.

        Parameters
        ----------
        model : Model
            The Keras model to explain.
        batch_size : int, optional
            How many occluded images to feed to the model in a single
            prediction batch.

        Returns
        -------
        array-like
            Numpy matrix representing the predictions for occluded images.
        """
        prob_matrix = np.zeros((self.output_height, self.output_width))
        predictions = list()

        for idx_start in range(0, len(self.occluded_subimages), batch_size):
            idx_end = min(len(self.occluded_subimages), idx_start + batch_size)
            batch = np.array([
                macrotile[0]
                for macrotile in self.occluded_subimages[idx_start:idx_end]]
            )
            predictions = predictions + [
                # TODO: [0] is T/F specific; class has to be chosen in advance
                x[0] for x in model.predict(batch, batch_size=batch.shape[0])
            ]

        for idx, prediction in enumerate(predictions):
            prob_matrix[
                self.occluded_subimages[idx][1], self.occluded_subimages[idx][2]
            ] = prediction
        return prob_matrix
