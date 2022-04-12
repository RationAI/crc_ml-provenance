# Standard Imports
from abc import ABC
from typing import NoReturn
from pathlib import Path

# Third-party Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LayerNormalization

# Local Imports
from rationai.training.base.models import Model
from rationai.utils.config import ConfigProto
from rationai.utils.class_handler import get_class
from rationai.utils.provenance import SummaryWriter

import logging
log = logging.getLogger('models')
sw_log = SummaryWriter.getLogger('provenance')


class KerasModel(ABC, Model):
    def __init__(self, config: ConfigProto, name: str):
        self.name = name
        self.config = config
        self.model = None

    def load_weights(self) -> NoReturn:
        if self.config.checkpoint is not None:
            log.info(f'Loading weights from: {self.config.checkpoint}')
            sw_log.set('model', 'checkpoint_file', value=str(Path(self.config.checkpoint).resolve()))
            self.model.load_weights(str(self.config.checkpoint)).expect_partial()

    def save_weights(self, output_path) -> NoReturn:
        self.model.save_weights(output_path, save_format='tf')

    def compile_model(self):
        raise NotImplementedError

    class Config(ConfigProto):
        def __init__(self, json_dict: dict):
            super().__init__(json_dict)
            self.seed = None
            self.checkpoint = None
            self.input_shape = None
            self.output_size = None
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
            self.seed = self.config.get('seed', np.random.randint(low=0, high=999999))
            sw_log.set('seed', 'model', value=self.seed)
            self.checkpoint = self.config.get('checkpoint', None)
            self.input_shape = tuple(self.config['input_shape'])
            self.output_size = self.config['output_size']

            components_config = self.config['components']
            configuration_config = self.config['configurations']


            # Activation Function
            self.output_activation_fn = get_class(
                components_config.get(
                    'output_activation_fn',
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
            if len(self.metric_configs) == 0:
                self.metric_configs = [dict()] * len(self.metric_classes)
            assert len(self.metric_configs) == len(self.metric_classes), f'Number of configurations different from number of metric classes. {len(self.metric_configs)} != {len(self.metric_classes)}'

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

class PretrainedNet(KerasModel):
    def __init__(self, config):
        super().__init__(config, 'PretrainedModel')
        self.model = self._build_model()
        self.load_weights()

    def _build_model(self):
        inp = Input(self.config.input_shape)
        pretrainelat_vec_size = self.config.convolution_network_class(
            **self.config.convolution_network_config, input_tensor=inp
        )
        pretrainelat_vec_size.trainable=True

        log.info(f'Building {pretrainelat_vec_size.name} model.')
        log.info(f'Model input size: {self.config.input_shape}')

        # Apply regularization on pretrained network
        if self.config.regularizer_class is not None:
            for layer in pretrainelat_vec_size.layers:
                # Kernel Regularization
                if hasattr(layer, 'kernel_regularizer'):
                    setattr(
                        layer,
                        'kernel_regularizer',
                        self.config.regularizer_class(
                            **self.config.regularizer_config
                        )
                    )

        out = pretrainelat_vec_size(inp)
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

    class Config(KerasModel.Config):
        def __init__(self, json_dict: dict):
            super().__init__(json_dict)
            self.dropout = None
            self.convolution_network = None

        def parse(self):
            super().parse()
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

class UNet(KerasModel):
    """UNet model for tissue segmentation."""

    def __init__(self, config):
        super().__init__(config, 'UNet')
        self.model = self._build_model()
        self.load_weights()

    def _build_model(self):
        inputs = Input(self.config.input_shape)

        reg = self.config.regularizer_class(**self.config.regularizer_config)

        c1 = Conv2D(64, (3, 3), activation=self.config.hidden_activation_fn, padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (inputs)
        c1 = Conv2D(64, (3, 3), activation=self.config.hidden_activation_fn, padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (c1)
        p1 = MaxPooling2D((2, 2)) (c1)

        c2 = Conv2D(128, (3, 3), activation=self.config.hidden_activation_fn, padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (p1)
        c2 = Conv2D(128, (3, 3), activation=self.config.hidden_activation_fn, padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (c2)
        p2 = MaxPooling2D((2, 2)) (c2)

        c3 = Conv2D(256, (3, 3), activation=self.config.hidden_activation_fn, padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (p2)
        c3 = Conv2D(256, (3, 3), activation=self.config.hidden_activation_fn, padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (c3)
        p3 = MaxPooling2D((2, 2)) (c3)

        c4 = Conv2D(512, (3, 3), activation=self.config.hidden_activation_fn, padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (p3)
        c4 = Conv2D(512, (3, 3), activation=self.config.hidden_activation_fn, padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (c4)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

        c5 = Conv2D(1024, (3, 3), activation=self.config.hidden_activation_fn, padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (p4)
        c5 = Conv2D(1024, (3, 3), activation=self.config.hidden_activation_fn, padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (c5)

        u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(512, (3, 3), activation=self.config.hidden_activation_fn, padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (u6)
        c6 = Conv2D(512, (3, 3), activation=self.config.hidden_activation_fn, padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (c6)

        u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(256, (3, 3), activation=self.config.hidden_activation_fn, padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (u7)
        c7 = Conv2D(256, (3, 3), activation=self.config.hidden_activation_fn, padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (c7)

        u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(128, (3, 3), activation=self.config.hidden_activation_fn, padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (u8)
        c8 = Conv2D(128, (3, 3), activation=self.config.hidden_activation_fn, padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (c8)

        u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(64, (3, 3), activation=self.config.hidden_activation_fn, padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (u9)
        c9 = Conv2D(64, (3, 3), activation=self.config.hidden_activation_fn, padding=self.config.hidden_padding, kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (c9)

        outputs = Conv2D(1, (1, 1), kernel_regularizer=reg, kernel_initializer=self.config.hidden_kernel_initializer_fn()) (c9)
        outputs = Activation(self.config.output_activation_fn)(outputs)
        full_model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        return full_model

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

    class Config(KerasModel.Config):
        def __init__(self, json_dict):
            super().__init__(json_dict)
            self.hidden_activation_fn = None
            self.hidden_padding = None
            self.hidden_kernel_initializer_fn = None

        def parse(self):
            super().parse()
            components_config = self.config['components']

            # Hidden Activation Function
            self.hidden_activation_fn = get_class(
                components_config.get(
                    'hidden_activation_fn',
                    None
                )
            )

            # Hidden Padding
            self.hidden_padding = components_config.get(
                    'hidden_padding',
                    'same'
            )

            # Hidden Kernel Initializer
            self.hidden_kernel_initializer_fn = get_class(
                components_config.get(
                    'hidden_kernel_initializer_fn',
                    None
                )
            )

class VisionTransformer(KerasModel):

    class MultiHeadSelfAttention(tf.keras.layers.Layer):
        def __init__(self, embed_dim, num_heads=8):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            if embed_dim % num_heads != 0:
                raise ValueError(
                    f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
                )
            self.projection_dim = embed_dim // num_heads
            self.query_dense = Dense(embed_dim)
            self.key_dense = Dense(embed_dim)
            self.value_dense = Dense(embed_dim)
            self.combine_heads = Dense(embed_dim)

        def attention(self, query, key, value):
            score = tf.matmul(query, key, transpose_b=True)
            dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
            scaled_score = score / tf.math.sqrt(dim_key)
            weights = tf.nn.softmax(scaled_score, axis=-1)
            output = tf.matmul(weights, value)
            return output, weights

        def separate_heads(self, x, batch_size):
            x = tf.reshape(
                x, (batch_size, -1, self.num_heads, self.projection_dim)
            )
            return tf.transpose(x, perm=[0, 2, 1, 3])

        def call(self, inputs):
            batch_size = tf.shape(inputs)[0]
            query = self.query_dense(inputs)
            key = self.key_dense(inputs)
            value = self.value_dense(inputs)
            query = self.separate_heads(query, batch_size)
            key = self.separate_heads(key, batch_size)
            value = self.separate_heads(value, batch_size)

            attention, weights = self.attention(query, key, value)
            attention = tf.transpose(attention, perm=[0, 2, 1, 3])
            concat_attention = tf.reshape(
                attention, (batch_size, -1, self.embed_dim)
            )
            output = self.combine_heads(concat_attention)
            return output

    class TransformerBlock(tf.keras.layers.Layer):
        def __init__(self, embed_dim, num_heads, mlp_hidden_size, dropout=0.1):
            super().__init__()
            self.att = VisionTransformer.MultiHeadSelfAttention(embed_dim, num_heads)
            self.mlp = tf.keras.Sequential(
                [
                    Dense(mlp_hidden_size, activation=tf.keras.activations.relu),
                    Dropout(dropout),
                    Dense(embed_dim),
                    Dropout(dropout),
                ]
            )
            self.layernorm1 = LayerNormalization(epsilon=1e-6)
            self.layernorm2 = LayerNormalization(epsilon=1e-6)
            self.dropout1 = Dropout(dropout)
            self.dropout2 = Dropout(dropout)

        def call(self, inputs, training):
            inputs_norm = self.layernorm1(inputs)
            attn_output = self.att(inputs_norm)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = attn_output + inputs

            out1_norm = self.layernorm2(out1)
            mlp_output = self.mlp(out1_norm)
            mlp_output = self.dropout2(mlp_output, training=training)
            return mlp_output + out1

    class EmbeddingLayer(tf.keras.layers.Layer):
        def __init__(self, name, shape):
            super().__init__(name=name)
            self.W = self.add_weight(name, shape=shape)

        def call(self, _):
            return self.W

    def __init__(self, config):
        super().__init__(config, 'VisualTransformer')
        self.model = self._build_model()
        self.load_weights()

    def _build_model(self):
        image_size = self.config.input_shape[0]
        if image_size % self.config.patch_size != 0 :
            raise Exception("Image size is not divisible by patch size")

        num_patches = (image_size // self.config.patch_size) ** 2
        self._patch_dim = self.config.input_shape[-1] * self.config.patch_size ** 2

        inp = Input(self.config.input_shape)

        pos_emb = VisionTransformer.EmbeddingLayer(name='pos_emb', shape=(1, num_patches + 1, self.config.lat_vec_size))
        class_emb = VisionTransformer.EmbeddingLayer(name='class_emb', shape=(1, 1, self.config.lat_vec_size))

        patch_proj = Dense(self.config.lat_vec_size, name = "patch_embedding")

        enc_layers = [
            VisionTransformer.TransformerBlock(
                self.config.lat_vec_size,
                self.config.num_heads,
                self.config.mlp_hidden_size,
                self.config.dropout
            ) for _ in range(self.config.num_blocks)
        ]
        mlp_head = tf.keras.Sequential(
            [
                LayerNormalization(epsilon=1e-6),
                Dense(self.config.mlp_hidden_size, activation=tf.keras.activations.relu),
                Dropout(self.config.dropout),
                Dense(self.config.output_size, activation=tf.keras.activations.sigmoid),
            ]
        )

        batch_size = tf.shape(inp)[0]
        patches = self.__extract_patches(inp)

        out = patch_proj(patches)

        class_emb.W = tf.broadcast_to(
            class_emb.W, [batch_size, 1, self.config.lat_vec_size]
        )
        out = tf.concat([class_emb.W, out], axis=1)
        out = out + pos_emb.W

        for layer in enc_layers:
            out = layer(out)

        # First (class token) is used for classification
        out = mlp_head(out[:, 0])
        model = tf.keras.Model(inp, out)
        return model

    def __extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.config.patch_size, self.config.patch_size, 1],
            strides=[1, self.config.patch_size, self.config.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self._patch_dim])
        return patches

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

    class Config(KerasModel.Config):
        def __init__(self, json_dict: dict):
            super().__init__(json_dict)
            self.patch_size = None
            self.num_blocks = None
            self.num_heads = None
            self.lat_vec_size = None
            self.mlp_hidden_size = None
            self.dropout = None

        def parse(self):
            super().parse()
            self.dropout = self.config.get('dropout', 0.0)
            self.patch_size = self.config['patch_size']
            self.num_blocks = self.config['num_blocks']
            self.num_heads = self.config['num_heads']
            self.lat_vec_size = self.config['lat_vec_size']
            self.mlp_hidden_size = self.config['mlp_hidden_size']
