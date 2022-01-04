import tensorflow as tf

class DiceLoss:
    def __init__(self, smooth=1e-6):
        self.name = f'dice_loss(smooth={smooth})'
        self.__name__ = 'dice_loss'
        self.smooth = smooth

    def __call__(self, targets, inputs):
        inputs = tf.keras.backend.flatten(inputs)
        targets = tf.keras.backend.flatten(targets)
        intersection = tf.tensordot(targets, inputs, axes=1)
        dice = (2*intersection + self.smooth) / (tf.keras.backend.sum(targets) + tf.keras.backend.sum(inputs) + self.smooth)
        return 1 - dice

    def __repr__(self):
        return self.name


class DiceBCELoss:
    def __init__(self, smooth=1e-6):
        self.name = f'Dice+BCE_Loss(smooth={smooth})'
        self.__name__ = 'dice_bce_loss'
        self.smooth = smooth

    def __call__(self, targets, inputs):
        inputs = tf.keras.backend.flatten(inputs)
        targets = tf.keras.backend.flatten(targets)
        BCE = tf.keras.losses.binary_crossentropy(targets, inputs)
        intersection = tf.tensordot(targets, inputs, axes=1)
        dice_loss = 1 - (2*intersection + self.smooth) / (tf.keras.backend.sum(targets) + tf.keras.backend.sum(inputs) + self.smooth)
        Dice_BCE = BCE + dice_loss
        return Dice_BCE

    def __repr__(self):
        return self.name