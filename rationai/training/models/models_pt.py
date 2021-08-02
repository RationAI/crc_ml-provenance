"""
PyTorch models and wrappers.
"""
import abc
import logging
from pathlib import Path
from typing import List, NoReturn, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as nn_func
import torchmetrics

from rationai.utils import pytorchutils
from rationai.utils.typealias import TorchRegularizer, TorchOptimGenerator

log = logging.getLogger('models_pt')


class PretrainedVGG16(nn.Module):
    """
    VGG16 variant convolutional model.
    """

    def __init__(self):
        super().__init__()
        # Block 1
        self.block1_conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block1_conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block1_pool = nn.MaxPool2d(2, stride=2, padding=0)

        # Block 2
        self.block2_conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block2_conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block2_pool = nn.MaxPool2d(2, stride=2, padding=0)

        # Block 3
        self.block3_conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block3_conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block3_conv3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block3_pool = nn.MaxPool2d(2, stride=2, padding=0)

        # Block 4
        self.block4_conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block4_conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block4_conv3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block4_pool = nn.MaxPool2d(2, stride=2, padding=0)

        # Block 5
        self.block5_conv1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block5_conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block5_conv3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block5_pool = nn.MaxPool2d(2, stride=2, padding=0)

        self.dropout = nn.Dropout(.5)
        self.dense = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        RAI_UNTESTED

        Parameters
        ----------
        x : torch.Tensor
            Input to the model.

        Return
        ------
        torch.Tensor
            A tensor representing a binary decision over the input samples.
        """
        # Block 1
        x = nn_func.relu(self.block1_conv1(x))
        x = nn_func.relu(self.block1_conv2(x))
        x = self.block1_pool(x)

        # Block 2
        x = nn_func.relu(self.block2_conv1(x))
        x = nn_func.relu(self.block2_conv2(x))
        x = self.block2_pool(x)

        # Block 3
        x = nn_func.relu(self.block3_conv1(x))
        x = nn_func.relu(self.block3_conv2(x))
        x = nn_func.relu(self.block3_conv3(x))
        x = self.block3_pool(x)

        # Block 4
        x = nn_func.relu(self.block4_conv1(x))
        x = nn_func.relu(self.block4_conv2(x))
        x = nn_func.relu(self.block4_conv3(x))
        x = self.block4_pool(x)

        # Block 5
        x = nn_func.relu(self.block5_conv1(x))
        x = nn_func.relu(self.block5_conv2(x))
        x = nn_func.relu(self.block5_conv3(x))
        x = self.block5_pool(x)

        # GlobalMaxPool
        x = nn_func.max_pool2d(x, kernel_size=x.size()[2:])

        x = self.dropout(x)
        # [batch, side_size, 1, 1] -> [batch, 512, 1]
        x = torch.squeeze(x, -1)
        # [batch, side_size, 1] -> [batch, 512]
        x = torch.squeeze(x, -1)
        x = torch.sigmoid(self.dense(x))
        # [batch, 1] -> [batch]
        x = torch.squeeze(x, -1)

        return x


class BaseTorchModel(abc.ABC):
    """
    Wrapper class for PyTorch Model.

    Handles instantiation of model, regularizer, optimizer, loss function and
    metrics.
    """

    def __init__(self, name: str, model_config: dict, checkpoint_dir: Path):
        self.name = name
        self.model_config = model_config
        self.model = None
        self.checkpoint_dir = checkpoint_dir

        # Auto load from config
        self.regularizer = self._get_regularizer()
        self.optimizer = self._get_optimizer()
        self.loss = self._get_loss_function()
        self.sequence_metrics = self._get_metrics()
        self.total_metrics = self._get_metrics(total=True)

        log.info(f'Building {self.name} model.')
        log.info(f"Model input size: {self.model_config['input_shape']}")

    def load_weights(self, checkpoint_fn: Union[str, Path]) -> NoReturn:
        """
        Load checkpoint weights for PyTorch model.

        RAI_UNTESTED

        Parameters
        ----------
        checkpoint_fn : Union[str, Path]
            Checkpoint filename. This is searched for in the directory provided
            upon `BaseTorchModel` initialization, or in the project directory
            if the first attempt fails.
        """

        if not checkpoint_fn:
            log.warning('No model checkpoint provided')
            return

        checkpoint = self.checkpoint_dir / checkpoint_fn
        if checkpoint.exists():
            # checkpoint is in checkpoint_dir or defined by absolute path
            log.info(f'Loading weights from: {str(checkpoint)}')
            self.model.load_state_dict(
                torch.load(checkpoint)['model_state_dict']
            )
        elif Path(checkpoint_fn).exists():
            # checkpoint found by relative path in project dir
            log.info(
                'Loading weights from a from project dir using relative path:'
                f' {str(checkpoint_fn)}'
            )
            self.model.load_state_dict(
                torch.load(checkpoint_fn)['model_state_dict']
            )
        else:
            log.warning(
                f'Provided checkpoint file name {checkpoint_fn} not found'
            )

    def test_mode(self) -> 'BaseTorchModel':
        """
        Load weights from test checkpoint file.

        Should be executed before experiment test phase.

        RAI_UNTESTED

        Return
        ------
        BaseTorchModel
            This `BaseTorchModel` with the weights set to the test checkpoint
            saved weights.
        """
        self.load_weights(self.model_config.get('test_checkpoint', None))
        return self

    def get_sequence_metrics(self) -> List[torchmetrics.Metric]:
        """
        Provide a list of sequence metrics currently used in the model.

        RAI_UNTESTED

        Return
        ------
        List[torchmetrics.Metric]
            A list of sequence metrics used in the model.
        """
        return self.sequence_metrics

    def get_total_metrics(self) -> List[torchmetrics.Metric]:
        """
        Provide a list of total metrics currently used in the model.

        RAI_UNTESTED

        Return
        ------
        List[torchmetrics.Metric]
            A list of total metrics used in the model.
        """
        if hasattr(self, 'total_metrics'):
            return self.total_metrics
        return []

    def _get_regularizer(self) -> Optional[TorchRegularizer]:
        """
        Prepare the regularizer to be used in model training from model config.

        RAI_UNTESTED

        Return
        -----
        Optional[TorchRegularizer]
            The regularization function to use during model training (see
            `typealias` for specification) or None, if regularizer name or
            config are invalid.
        """
        try:
            name: str = self.model_config['regularizer']['class_name']
        except KeyError:
            log.info('No regularizer info provided, will not be used.')
            return None

        try:
            config: dict = self.model_config['regularizer']['config']
        except KeyError:
            log.warning('No regularizer config provided, will not be used.')
            return None

        regularizer = pytorchutils.get_pytorch_regularizer(name, config)

        if regularizer is None:
            log.warning(f'Unknown regularizer: {name}, will not be used.')
        else:
            log.info(f'Using regularizer: {name}')

        return regularizer

    def _get_optimizer(self) -> Optional[TorchOptimGenerator]:
        """
        Prepare the optimizer to be used in model training from model config.

        RAI_UNTESTED

        Return
        -----
        Optional[TorchRegularizer]
            The optimizer generation function to use during model training (see
            `typealias` for specification) or None, if optimizer name or
            config are invalid.
        """
        try:
            name = self.model_config['optimizer']['class_name']
        except KeyError:
            log.warning('No optimizer info provided, will not be used.')
            return None

        try:
            config = self.model_config['optimizer']['config']
        except KeyError:
            log.warning('No optimizer config provided, will not be used.')
            return None

        optimizer = pytorchutils.get_pytorch_optimizer(name, config)

        if optimizer is None:
            log.warning(f'Unknown regularizer: {name}, will not be used.')
        else:
            log.info(f'Using regularizer: {name}')

        return optimizer

    def _get_loss_function(self) -> Optional[torch.nn.modules.loss._Loss]:
        """
        Prepare the loss function to use in model training from model config.

        RAI_UNTESTED

        Return
        ------
        Optional[torch.nn.modules.loss._Loss]
            The loss function to use during model training, None if loss name
            is not provided or is unknown.
        """
        try:
            name = self.model_config['loss']['class_name']
        except KeyError:
            log.warning('No loss info provided, will not be used.')
            return None

        loss = pytorchutils.get_pytorch_loss(name)

        if loss is None:
            log.warning(f'Unknown loss: {name}, will not be used.')
        else:
            log.info(f'Using loss: {name}')

        return loss

    def _get_metrics(self, total=False) -> List[torchmetrics.Metric]:
        """
        Loads PyTorch metrics from config to use in training and evaluation.

        RAI_UNTESTED

        Parameters
        ----------
        total : bool
            Whether to load total metrics (which accumulate throughout the
            process) or sequence metrics (which get reset after each epoch.)

        Return
        ------
        List[torchmetrics.Metric]
            A list of metrics parsed from the config.
        """
        config_key = 'total_metrics' if total else 'sequence_metrics'
        config = self.model_config.get(config_key, [])

        metrics = []
        if config:
            for metric_info in config:
                metric_name = metric_info['class_name']
                metric = pytorchutils.get_pytorch_metric(metric_name)
                if metric is not None:
                    metrics.append(metric)
                else:
                    log.warning(f'Unknown sequence metric name: {metric_name}')

        return metrics
