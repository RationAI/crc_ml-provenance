import numpy as np

from rationai.training.base.evaluators import Evaluator
from rationai.utils.config import ConfigProto

class TruePositives(Evaluator):
    def __init__(self, config: ConfigProto):
        super().__init__(config)
        self.name = self.name or "True Positives"

    def reset_state(self):
        self.TP = 0

    def update_state_fn(self, y_pred: np.ndarray, y_true: np.ndarray):
        assert len(y_pred) == len(y_true), \
            f'"y_pred" and "y_true" must have same number of elements  \
              ({len(y_pred)} == {len(y_true)}).'

        y_pred = (
            np.array(y_pred, dtype=float) > self.config.threshold
        ).astype(bool)
        y_true = np.array(y_true, dtype=bool)
        self.TP += np.sum(y_true & y_pred)

    def result(self):
        return self.TP

class TrueNegatives(Evaluator):
    def __init__(self, config: ConfigProto):
        super().__init__(config)
        self.name = self.name or "True Negatives"

    def reset_state(self):
        self.TN = 0

    def update_state_fn(self, y_pred: np.ndarray, y_true: np.ndarray):
        assert len(y_pred) == len(y_true), \
            f'"y_pred" and "y_true" must have same number of elements  \
              ({len(y_pred)} == {len(y_true)}).'

        y_pred = (
            np.array(y_pred, dtype=float) > self.config.threshold
        ).astype(bool)
        y_true = np.array(y_true, dtype=bool)
        self.TN += np.sum(~y_true & ~y_pred)

    def result(self):
        return self.TN

class FalsePositives(Evaluator):
    def __init__(self, config: ConfigProto):
        super().__init__(config)
        self.name = self.name or "False Positives"

    def reset_state(self):
        self.FP = 0

    def update_state_fn(self, y_pred: np.ndarray, y_true: np.ndarray):
        assert len(y_pred) == len(y_true), \
            f'"y_pred" and "y_true" must have same number of elements  \
              ({len(y_pred)} == {len(y_true)}).'

        y_pred = (
            np.array(y_pred, dtype=float) > self.config.threshold
        ).astype(bool)
        y_true = np.array(y_true, dtype=bool)
        self.FP += np.sum(~y_true & y_pred)

    def result(self):
        return self.FP

class FalseNegatives(Evaluator):
    def __init__(self, config: ConfigProto):
        super().__init__(config)
        self.name = self.name or "False Negatives"

    def reset_state(self):
        self.FN = 0

    def update_state_fn(self, y_pred: np.ndarray, y_true: np.ndarray):
        assert len(y_pred) == len(y_true), \
            f'"y_pred" and "y_true" must have same number of elements  \
              ({len(y_pred)} == {len(y_true)}).'

        y_pred = (
            np.array(y_pred, dtype=float) > self.config.threshold
        ).astype(bool)
        y_true = np.array(y_true, dtype=bool)
        self.FN += np.sum(y_true & ~y_pred)

    def result(self):
        return self.FN

class BinaryAccuracy(Evaluator):
    def __init__(self, config: ConfigProto):
        super().__init__(config)
        self.name = self.name or "Binary Accuracy"

    def reset_state(self):
        self.TPTN = 0
        self.TPTNFPFN = 0

    def update_state_fn(self, y_pred: np.ndarray, y_true: np.ndarray):
        assert len(y_pred) == len(y_true), \
            f'"y_pred" and "y_true" must have same number of elements  \
              ({len(y_pred)} == {len(y_true)}).'

        y_pred = (
            np.array(y_pred, dtype=float) > self.config.threshold
        ).astype(bool)
        y_true = np.array(y_true, dtype=bool)
        self.TPTN += np.sum(y_pred == y_true)
        self.TPTNFPFN += len(y_pred)

    def result(self):
        if self.TPTNFPFN == 0:
            return 1.0
        return self.TPTN / self.TPTNFPFN

class Precision(Evaluator):
    def __init__(self, config: ConfigProto):
        super().__init__(config)
        self.name = self.name or "Precision"

    def reset_state(self):
        self.TP = 0
        self.TPFP = 0

    def update_state_fn(self, y_pred: np.ndarray, y_true: np.ndarray):
        assert len(y_pred) == len(y_true), \
            f'"y_pred" and "y_true" must have same number of elements  \
              ({len(y_pred)} == {len(y_true)}).'

        y_pred = (
            np.array(y_pred, dtype=float) > self.config.threshold
        ).astype(bool)
        y_true = np.array(y_true, dtype=bool)
        self.TP += np.sum(y_true & y_pred)
        self.TPFP += np.sum(y_pred)

    def result(self):
        if self.TPFP == 0:
            return 1.0
        return self.TP / self.TPFP

class Recall(Evaluator):
    def __init__(self, config: ConfigProto):
        super().__init__(config)
        self.name = self.name or "Recall"

    def reset_state(self):
        self.TP = 0
        self.TPFN = 0

    def update_state_fn(self, y_pred: np.ndarray, y_true: np.ndarray):
        assert len(y_pred) == len(y_true), \
            f'"y_pred" and "y_true" must have same number of elements  \
              ({len(y_pred)} == {len(y_true)}).'

        y_pred = (
            np.array(y_pred, dtype=float) > self.config.threshold
        ).astype(bool)
        y_true = np.array(y_true, dtype=bool)
        self.TP += np.sum(y_true & y_pred)
        self.TPFN += np.sum(y_true)

    def result(self):
        if self.TPFN == 0:
            return 1.0
        return self.TP / self.TPFN

class F1(Evaluator):
    def __init__(self, config: ConfigProto):
        super().__init__(config)
        self.name = self.name or "F1"

    def reset_state(self):
        self.TP = 0
        self.FPFN = 0

    def update_state_fn(self, y_pred: np.ndarray, y_true: np.ndarray):
        assert len(y_pred) == len(y_true), \
            f'"y_pred" and "y_true" must have same number of elements  \
              ({len(y_pred)} == {len(y_true)}).'

        y_pred = (
            np.array(y_pred, dtype=float) > self.config.threshold
        ).astype(bool)
        y_true = np.array(y_true, dtype=bool)

        self.TP += np.sum(y_true & y_pred)
        self.FPFN += len(y_pred) - np.sum(y_true == y_pred)

    def result(self):
        if (self.TP + 0.5*self.FPFN) == 0:
            return 1.0
        return self.TP / (self.TP + 0.5*self.FPFN)

class Specificity(Evaluator):
    def __init__(self, config: ConfigProto):
        super().__init__(config)
        self.name = self.name or "Specificity"

    def reset_state(self):
        self.TN = 0
        self.TNFP = 0

    def update_state_fn(self, y_pred: np.ndarray, y_true: np.ndarray):
        assert len(y_pred) == len(y_true), \
            f'"y_pred" and "y_true" must have same number of elements  \
              ({len(y_pred)} == {len(y_true)}).'

        y_pred = (
            np.array(y_pred, dtype=float) > self.config.threshold
        ).astype(bool)
        y_true = np.array(y_true, dtype=bool)
        self.TN += np.sum(~y_true & ~y_pred)
        self.TNFP += np.sum(~y_true)

    def result(self):
        if self.TNFP == 0:
            return 1.0
        return self.TN / self.TNFP

