import tensorflow as tf

class ThresholdedMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes, name=None, dtype=None, threshold=0.5):
        self.threshold=threshold
        return super().__init__(num_classes, name, dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, y_pred>self.threshold, sample_weight=sample_weight)