from __future__ import annotations

import abc
import logging
import tensorflow as tf
import types
from copy import deepcopy
from types import MethodType
from pathlib import Path
from tensorflow.keras import layers as klay
from typing import List
from typing import NoReturn
from typing import Union

from rationai.training.models.model_utils import test_step
from rationai.training.models.model_utils import make_test_function
from rationai.training.models.model_utils import evaluate
from rationai.utils import load_from_module
from rationai.training.losses import load_loss
from rationai.training.metrics import load_metrics

log = logging.getLogger('models')


def load_keras_model(model_config: dict, ckpt_dir: Path):
    """Returns a custom Keras model wrapper.

    Args:
    model_config: dict
        Model configuration parameters.

    ckpt_dir: Path
        Path to a weight checkpoint directory.
    """

    class_name = model_config['class_name']
    kwargs = {'model_config': model_config, 'ckpt_dir': ckpt_dir}
    path = f'{__name__}.{class_name}'
    return load_from_module(path, **kwargs)


def load_keras_application(identifier: dict):
    """Returns pretrained Keras model from tf.keras.applications.

    Args:
    identifier: dict
        Keras identifier.
        ('include_top' is always False)
    """

    identifier['config']['include_top'] = False
    path = f"tf.k.applications.{identifier['class_name']}"
    return load_from_module(path, **identifier['config'])


class BaseModel(abc.ABC):
    """Wrapper class for Keras Model.
    Handles instantiation of model, regularizer, optimizer,
    loss function and metrics"""
    def __init__(self, name, model_config, ckpt_dir: Path):
        self.name = name
        self.model_config = model_config
        self.model = None
        self.ckpt_dir = ckpt_dir

        # Auto load from config
        self.regularizer = self._get_regularizer()
        self.optimizer = self._get_optimizer()
        self.loss = self._get_loss_function()
        self.metrics = self._get_sequence_metrics()
        self.total_metrics = self._get_total_metrics()

        log.info(f'Building {self.name} model.')
        log.info(f"Model input size: {self.model_config['input_shape']}")

    def load_weights(self, ckpt_fn: Union[str, Path]) -> NoReturn:
        """Tries to load checkpoint weights for Keras model.
        Automatically chooses either train or test chechkpoint"""

        if ckpt_fn in [None, '']:
            log.info('No model checkpoint provided')
            return

        ckpt = self.ckpt_dir / ckpt_fn
        if ckpt.exists():  # ckpt is in ckpt_dir or defined by absolute path
            log.info(f'Loading weights from: {str(ckpt)}')
            self.model.load_weights(str(ckpt))
        elif Path(ckpt_fn).exists():  # ckpt found by relative path in project dir
            log.info('Loading weights from a from project dir '
                     f'using relative path: {str(ckpt_fn)}')
            self.model.load_weights(str(ckpt_fn))
        else:
            log.info(f'Provided ckpt file name {ckpt_fn} not found')

    def compile(self) -> NoReturn:
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

    def test_mode(self) -> BaseModel:
        """Needs to be executed before experiment test phase.
        If possible, loads test ckpt weights, total_metrics and recompiles model."""
        self.metrics += self.total_metrics
        self.load_weights(self.model_config.get('test_checkpoint', None))
        self.compile()
        return self

    def override_predict_evaluate(self) -> NoReturn:
        """Overrides Keras Model's methods
        to run predict and evaluate as one.
        making predictions and evaluating metrics as a single step.
        """
        self.model.test_step = types.MethodType(test_step, self.model)
        self.model.make_test_function = types.MethodType(
            make_test_function, self.model)
        self.model.evaluate = types.MethodType(evaluate, self.model)

    def get_metrics(self) -> list[tf.keras.metrics.Metric]:
        """Returns a list of all metrics which are currently used in the model"""
        return self.metrics

    def get_total_metrics(self) -> List[tf.keras.metrics.Metric]:
        if hasattr(self, 'total_metrics'):
            return self.total_metrics
        return []

    def _get_regularizer(self) -> tf.keras.regularizers.Regularizer:
        reg = tf.keras.regularizers.get(self.model_config['regularizer'])
        log.info(f'Using regularizer: {reg.get_config()}')
        return reg

    def _get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        optim = tf.keras.optimizers.get(self.model_config['optimizer'])
        log.info(f"Using optimizer: {optim.get_config()['name']}")
        return optim

    def _get_loss_function(self) -> tf.keras.losses.Loss:  # Union with BaseLoss
        loss = load_loss(self.model_config['loss'])
        log.info(f'Using loss function: {loss.name}')
        return loss

    def _totalify_metrics(self, metrics) -> NoReturn:
        """Given metrics will no longer reset their states on epoch end"""

        def reset_states(self):
            """Override prevents metric state reset on epoch end."""
            pass

        for m in metrics:
            m.reset_states = MethodType(reset_states, m)
            m._name = f'total-{m.name}'

    def _get_sequence_metrics(self) -> List[tf.keras.metrics.Metric]:
        """Loads "per-slide" (per-sequence) metrics from config"""
        config = self.model_config.get('sequence_metrics')
        if config:
            return load_metrics(config)
        return []

    def _get_total_metrics(self) -> List[tf.keras.metrics.Metric]:
        """Loads "per-dataset" metrics from config"""
        metrics = load_metrics(self.model_config.get('total_metrics', []))
        self._totalify_metrics(metrics)
        return metrics


class PretrainedModel(BaseModel):
    """Keras based model for classification."""
    def __init__(self, model_config, ckpt_dir):
        super().__init__('PretrainedModel', model_config, ckpt_dir)
        self.model = self._build_model()
        self.load_weights(model_config.get('train_checkpoint', None))
        self.compile()

    def _build_model(self):
        inp = klay.Input(self.model_config['input_shape'])
        pretrained_id = deepcopy(self.model_config['keras_application'])
        pretrained_id['config']['input_tensor'] = inp
        pretrained_model = load_keras_application(pretrained_id)
        pretrained_model.trainable = self.model_config['trainable']

        # TODO: could be moved to BaseModel?
        if self.regularizer is not None:
            for layer in pretrained_model.layers:
                for attr in ['kernel_regularizer']:
                    if hasattr(layer, attr):
                        setattr(layer, attr, self.regularizer)

        out = pretrained_model(inp)
        out = klay.Dropout(self.model_config['dropout'])(out)

        if self.model_config['output_bias'] is None:
            output_bias = 'zeros'
        else:
            output_bias = tf.keras.initializers.Constant(self.model_config['output_bias'])

        out = klay.Dense(self.model_config['output_size'],
                         kernel_regularizer=self.regularizer,
                         bias_initializer=output_bias,
                         activation='sigmoid')(out)

        model = tf.keras.Model(inp, out)

        return model


class UNet(BaseModel):
    """UNet model for tissue segmentation."""

    def __init__(self, model_config, ckpt_dir):
        super().__init__('UNet', model_config, ckpt_dir)
        self.model = self._build_model()
        self.load_weights(model_config.get('train_checkpoint', None))
        self.compile()

    def _build_model(self):
        inputs = klay.Input(self.model_config['input_shape'])

        reg = self.regularizer

        c1 = klay.Conv2D(64, (3, 3), activation='selu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (inputs)
        c1 = klay.Conv2D(64, (3, 3), activation='selu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (c1)
        p1 = klay.MaxPooling2D((2, 2)) (c1)

        c2 = klay.Conv2D(128, (3, 3), activation='selu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (p1)
        c2 = klay.Conv2D(128, (3, 3), activation='selu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (c2)
        p2 = klay.MaxPooling2D((2, 2)) (c2)

        c3 = klay.Conv2D(256, (3, 3), activation='selu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (p2)
        c3 = klay.Conv2D(256, (3, 3), activation='selu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (c3)
        p3 = klay.MaxPooling2D((2, 2)) (c3)

        c4 = klay.Conv2D(512, (3, 3), activation='selu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (p3)
        c4 = klay.Conv2D(512, (3, 3), activation='selu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (c4)
        p4 = klay.MaxPooling2D(pool_size=(2, 2)) (c4)

        c5 = klay.Conv2D(1024, (3, 3), activation='selu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (p4)
        c5 = klay.Conv2D(1024, (3, 3), activation='selu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (c5)

        u6 = klay.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (c5)
        u6 = klay.concatenate([u6, c4])
        c6 = klay.Conv2D(512, (3, 3), activation='selu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (u6)
        c6 = klay.Conv2D(512, (3, 3), activation='selu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (c6)

        u7 = klay.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (c6)
        u7 = klay.concatenate([u7, c3])
        c7 = klay.Conv2D(256, (3, 3), activation='selu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (u7)
        c7 = klay.Conv2D(256, (3, 3), activation='selu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (c7)

        u8 = klay.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (c7)
        u8 = klay.concatenate([u8, c2])
        c8 = klay.Conv2D(128, (3, 3), activation='selu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (u8)
        c8 = klay.Conv2D(128, (3, 3), activation='selu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (c8)

        u9 = klay.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (c8)
        u9 = klay.concatenate([u9, c1], axis=3)
        c9 = klay.Conv2D(64, (3, 3), activation='selu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (u9)
        c9 = klay.Conv2D(64, (3, 3), activation='selu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal') (c9)

        outputs = klay.Conv2D(1, (1, 1), kernel_regularizer=reg, kernel_initializer='he_normal') (c9)
        outputs = klay.Activation('sigmoid')(outputs)
        outputs = klay.Reshape((-1,1))(outputs)
        full_model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        return full_model


class FCN8(BaseModel):
    """FCN-8 model for tissue segmentation"""

    def __init__(self, model_config, ckpt_dir):
        super().__init__('FCN-8', model_config, ckpt_dir)
        self.model = self._build_model()
        self.load_weights(model_config.get('train_checkpoint', None))
        self.compile()

    def _build_model(self):
        inputs = klay.Input(self.model_config['input_shape'])

        reg = self.regularizer

        c1 = klay.Conv2D(64, (4, 4), activation='relu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal', name='Conv-64-1') (inputs)
        c1 = klay.Conv2D(64, (4, 4), activation='relu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal', name='Conv-64-2') (c1)
        p1 = klay.MaxPooling2D((2, 2), name='Pool1') (c1)
        d1 = klay.Dense(1, activation='relu', name='Skip1')(p1)

        c2 = klay.Conv2D(128, (4, 4), activation='relu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal', name='Conv-128-1') (p1)
        c2 = klay.Conv2D(128, (4, 4), activation='relu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal', name='Conv-128-2') (c2)
        p2 = klay.MaxPooling2D((2, 2), name='Pool2') (c2)
        d2 = klay.Dense(1, activation='relu', name='Skip2')(p2)

        c3 = klay.Conv2D(256, (4, 4), activation='relu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal', name='Conv-256-1') (p2)
        c3 = klay.Conv2D(256, (4, 4), activation='relu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal', name='Conv-256-2') (c3)
        c3 = klay.Conv2D(256, (4, 4), activation='relu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal', name='Conv-256-3') (c3)
        p3 = klay.MaxPooling2D((2, 2), name='Pool3') (c3)
        d3 = klay.Dense(1, activation='relu', name='Skip3')(p3)

        c4 = klay.Conv2D(512, (4, 4), activation='relu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal', name='Conv-512-1') (p3)
        c4 = klay.Conv2D(512, (4, 4), activation='relu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal', name='Conv-512-2') (c4)
        c4 = klay.Conv2D(512, (4, 4), activation='relu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal', name='Conv-512-3') (c4)
        p4 = klay.MaxPooling2D(pool_size=(2, 2), name='Pool4') (c4)

        c5 = klay.Conv2D(4096, (4, 4), activation='relu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal', name='Conv-4096-1') (p4)
        c5 = klay.Conv2D(4096, (4, 4), activation='relu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal', name='Conv-4096-2') (c5)

        u5 = klay.Conv2DTranspose(1, (4, 4), strides=(2, 2), activation='relu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal', name='Deconv1')(c5)
        u6 = klay.Conv2DTranspose(1, (4, 4), strides=(2, 2), activation='relu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal', name='Deconv2')(u5 + d3)
        u7 = klay.Conv2DTranspose(1, (4, 4), strides=(2, 2), activation='relu', padding='same', kernel_regularizer=reg, kernel_initializer='he_normal', name='Deconv3')(u6 + d2)
        u8 = klay.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', kernel_regularizer=reg, kernel_initializer='he_normal', name='Deconv4')(u7 + d1)
        outputs = klay.Activation('sigmoid')(u8)
        outputs = klay.Reshape((-1,1))(outputs)
        full_model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        return full_model
