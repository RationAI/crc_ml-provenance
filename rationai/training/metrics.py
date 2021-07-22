import numpy as np
import tensorflow.keras.metrics as tf_metrics
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops import math_ops, init_ops
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras import backend as K
from typing import List

from rationai.utils import load_from_module
from rationai.utils import join_module_path


def load_metrics(metrics_config: List[dict]):
    """Returns a list of deserialized custom or Keras metrics."""
    return [load_keras_metric(m) for m in metrics_config]


def load_keras_metric(identifier: dict):
    """Returns a single metric specified by dict identifier.

    Args:
    identifier: dict
        Keras-like identifier.
        {
            "class_name": "<class_name>",
            "config": {<kwargs>}
        }
    """

    class_name = identifier['class_name']
    config = identifier['config'] if 'config' in identifier else {}

    path = join_module_path(__name__, class_name) or f'tf.k.metrics.{class_name}'
    return load_from_module(path, **config)


class F1(tf_metrics.Metric):
    """F1 score metric"""
    def __init__(self, thresholds=None, top_k=None, class_id=None, name=None, dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold)
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)

    def result(self):
        p = math_ops.div_no_nan(self.true_positives,
                                self.true_positives + self.false_positives)
        r = math_ops.div_no_nan(self.true_positives,
                                self.true_positives + self.false_negatives)
        result = math_ops.div_no_nan(2 * (p * r), (p + r))
        return result[0] if len(self.thresholds) == 1 else result

    def reset_states(self):
        num_thresholds = len(to_list(self.thresholds))
        K.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        config = {
            'thresholds': self.init_thresholds,
            'top_k': self.top_k,
            'class_id': self.class_id
        }
        base_config = super(F1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MeanIoUWrapper(tf_metrics.MeanIoU):
    def __init__(self, num_classes, name=None, dtype=None, threshold=0.5):
        self.threshold = threshold
        return super().__init__(num_classes, name, dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(
            y_true,
            y_pred > self.threshold,
            sample_weight=sample_weight)


class Sensitivity(tf_metrics.Recall):
    """Recall wrapper for naming consistency (Specificity & Sensitivity)"""
    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name='sensitivity',
                 dtype=None):
        super().__init__(thresholds, top_k, class_id, name, dtype)


# SensitivitySpecificityBase is not part of API -> cannot be used
# class Specificity(tf_metrics.SensitivitySpecificityBase):
class Specificity(tf_metrics.Recall):
    """Specificity metric: computes TN / (TN+FP)"""
    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name='specificity',
                 dtype=None):
        super().__init__(thresholds, top_k, class_id, name, dtype)
        delattr(self, 'true_positives')
        delattr(self, 'false_negatives')

        self.true_negatives = self.add_weight(
            'true_negatives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)

    def result(self):
        result = math_ops.div_no_nan(
            self.true_negatives,
            self.true_negatives + self.false_positives)
        return result[0] if len(self.thresholds) == 1 else result
