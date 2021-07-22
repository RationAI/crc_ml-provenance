import tensorflow as tf
import tensorflow.keras.backend as K
from abc import ABC, abstractmethod

from rationai.utils import load_from_module
from rationai.utils import join_module_path


def load_loss(identifier):
    """Returns a deserialized custom or Keras loss functions.
W
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

    path = join_module_path(__name__, class_name) or f'tf.k.losses.{class_name}'
    return load_from_module(path, **config)


class BaseLoss(ABC):
    def __init__(self, name):
        self.name = name
        self.__name__ = name

    @abstractmethod
    def __call__(self, targets, inputs):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class DiceLoss(BaseLoss):
    def __init__(self, smooth=1e-6):
        super().__init__('dice_loss')
        self.smooth = smooth

    def __call__(self, targets, inputs):
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        intersection = tf.tensordot(targets, inputs, axes=1)
        dice = (2 * intersection + self.smooth) / (K.sum(targets) + K.sum(inputs) + self.smooth)
        return 1 - dice

    def __repr__(self):
        return f'{self.name}(smooth={self.smooth})'


class DiceBCELoss(BaseLoss):
    def __init__(self, smooth=1e-6):
        super().__init__('dice_bce_loss')
        self.smooth = smooth

    def __call__(self, targets, inputs):
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        bce = tf.keras.losses.binary_crossentropy(targets, inputs)
        intersection = tf.tensordot(targets, inputs, axes=1)
        dice_loss = 1 - (2 * intersection + self.smooth) / (K.sum(targets) + K.sum(inputs) + self.smooth)
        dice_bce = bce + dice_loss
        return dice_bce

    def __repr__(self):
        return f'{self.name}(smooth={self.smooth})'


class IoULoss(BaseLoss):
    def __init__(self, smooth=1e-6):
        super().__init__('iou_loss')
        self.smooth = smooth

    def __call__(self, targets, inputs):
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        intersection = K.sum(K.dot(targets, inputs))
        total = K.sum(targets) + K.sum(inputs)
        union = total - intersection
        IoU = (intersection + self.smooth) / (union + self.smooth)
        return 1 - IoU

    def __repr__(self):
        return f'{self.name}(smooth={self.smooth})'


class FocalLoss(BaseLoss):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__('focal_loss')
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, targets, inputs):
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        bce = K.binary_crossentropy(targets, inputs)
        bce_exp = K.exp(-bce)
        focal_loss = K.mean(self.alpha * K.pow((1 - bce_exp), self.gamma) * bce)
        return focal_loss

    def __repr__(self):
        return f'{self.name}(alpha={self.alpha}, gamma={self.gamma})'


class TverskyLoss(BaseLoss):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super().__init__('tversky_loss')
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def __call__(self, targets, inputs):
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        TP = K.sum((inputs * targets))
        FP = K.sum(((1 - targets) * inputs))
        FN = K.sum((targets * (1 - inputs)))
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky

    def __repr__(self):
        return f'{self.name}(alpha={self.alpha}, beta={self.beta}, smooth={self.smooth})'


class FocalTverskyLoss(BaseLoss):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1, smooth=1e-6):
        super().__init__('focal_tversky_loss')
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def __call__(self, targets, inputs):
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        TP = K.sum((inputs * targets))
        FP = K.sum(((1 - targets) * inputs))
        FN = K.sum((targets * (1 - inputs)))
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_tversky = K.pow((1 - tversky), self.gamma)
        return focal_tversky

    def __repr__(self):
        return f'{self.name}(alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}, smooth={self.smooth})'


class ComboLoss(BaseLoss):
    def __init__(self, alpha=0.5, ce_ratio=0.5):
        super().__init__('combo_loss')
        self.alpha = alpha
        self.ce_ratio = ce_ratio

    def __call__(self, targets, inputs):
        # NOTE: undefined variables were already present
        targets = K.flatten(targets)
        inputs = K.flatten(inputs)
        intersection = K.sum(targets * inputs)
        dice = (2. * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
        inputs = K.clip(inputs, e, 1.0 - e)
        out = - (self.alpha * ((targets * K.log(inputs)) + ((1 - self.alpha) * (1.0 - targets) * K.log(1.0 - inputs))))
        weighted_ce = K.mean(out, axis=-1)
        combo = (self.ce_ratio * weighted_ce) - ((1 - self.ce_ratio) * dice)
        return combo

    def __repr__(self):
        return f'{self.name}(alpha={self.alpha}, ce_ratio={self.ce_ratio})'
