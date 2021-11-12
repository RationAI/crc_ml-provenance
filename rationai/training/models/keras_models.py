# Standard Imports
import logging
from abc import ABC
from typing import NoReturn

# Third-party Imports
import tensorflow as tf

# Local Imports
from rationai.training.base.models import Model
from rationai.utils.class_handler import get_class
from rationai.utils.config import ConfigProto
from tensorflow.keras.layers import Dense, Dropout, Input

log = logging.getLogger('models')


class KerasModel(ABC, Model):
    def __init__(self, config: ConfigProto, name: str):
        self.name = name
        self.config = config
        self.model = None

    def load_weights(self) -> NoReturn:
        if self.config.checkpoint is not None:
            log.info(f'Loading weights from: {self.config.checkpoint}')
            self.model.load_weights(str(self.config.checkpoint))

    def compile_model(self):
        raise NotImplementedError

class PretrainedNet(KerasModel):
    def __init__(self, config):
        super().__init__(config, 'PretrainedModel')
        self.model = self._build_model()
        self.load_weights()

    def _build_model(self):
        inp = Input(self.config.input_shape)
        pretrained_model = self.config.convolution_network_class(
            **self.config.convolution_network_config, input_tensor=inp
        )
        pretrained_model.trainable=True

        log.info(f'Building {pretrained_model.name} model.')
        log.info(f'Model input size: {self.config.input_shape}')

        # Apply regularization on pretrained network
        if self.config.regularizer_class is not None:
            for layer in pretrained_model.layers:
                if hasattr(layer, 'kernel_regularizer'):
                    setattr(
                        layer,
                        'kernel_regularizer',
                        self.config.regularizer_class(
                            **self.config.regularizer_config
                        )
                    )

        out = pretrained_model(inp)
        out = Dropout(self.config.dropout)(out)
        out = Dense(
            self.config.output_size,
            kernel_regularizer=self.config.regularizer_class(
                **self.config.regularizer_config
            ),
            activation=self.config.output_activation_fn)(out)
        model = tf.keras.Model(inp, out)
        return model

    def compile_model(self):
        log.info(f'Using {self.config.optimizer_class.__name__} as optimizer.')
        metrics = [metric_cls(**metric_cfg) for metric_cls, metric_cfg in
                   zip(self.config.metric_classes, self.config.metric_configs)]
        self.model.compile(
            loss=self.config.loss_class(**self.config.loss_config),
            metrics=metrics,
            optimizer=self.config.optimizer_class(
                **self.config.optimizer_config
            )
        )

    class Config(ConfigProto):
        def __init__(self, json_dict: dict):
            super().__init__(json_dict)
            self.checkpoint = None
            self.input_shape = None
            self.output_size = None
            self.dropout = None
            self.convolution_network = None
            self.output_activation_fn = None

            self.optimizer_class = None
            self.optimizer_config = None

            self.metric_classes = None
            self.metric_configs = None

            self.loss_class = None
            self.loss_config = None

            self.regularizer_class = None
            self.regularizer_config = None


        def parse(self):
            self.checkpoint = self.config.get('checkpoint', None)
            self.input_shape = tuple(self.config['input_shape'])
            self.output_size = self.config['output_size']
            self.dropout = self.config.get('dropout', 0.0)

            components_config = self.config['components']
            configuration_config = self.config['configurations']

            # Convolution Network
            self.convolution_network_class = get_class(
                components_config.get(
                    'convolution_network',
                    'tensorflow.keras.applications.VGG16'
                )
            )
            self.convolution_network_config = configuration_config.get(
                'convolution_network',
                {'include_top': False, 'weights': 'imagenet', 'pooling': 'max'}
            )

            # Activation Function
            self.output_activation_fn = get_class(
                components_config.get(
                    'output_activation',
                    'tensorflow.keras.activations.linear'
                )
            )

            # Optimizer
            self.optimizer_class = get_class(
                components_config.get(
                    'optimizer',
                    'tensorflow.keras.optimizers.Adam'
                )
            )
            self.optimizer_config = configuration_config.get(
                'optimizer',
                dict()
            )

            # Metrics
            self.metric_classes = [get_class(metric_module)
                for metric_module in components_config.get(
                    'metrics',
                    ['tensorflow.keras.metrics.BinaryAccuracy']
                )
            ]
            self.metric_configs = configuration_config.get(
                'metrics',
                [dict()] * len(self.metric_classes)
            )

            # Loss
            self.loss_class = get_class(
                components_config.get(
                    'loss',
                    'tensorflow.keras.losses.BinaryCrossentropy'
                )
            )
            self.loss_config = configuration_config.get(
                'loss',
                dict()
            )

            # Regularizers
            self.regularizer_class = get_class(
                components_config.get(
                    'regularizer',
                    None
                )
            )
            self.regularizer_config = configuration_config.get(
                'regularizer',
                dict()
            )
