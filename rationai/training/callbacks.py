import numpy as np
from pathlib import Path
import tensorflow as tf
from typing import List
from typing import Type

from rationai.utils import (
    join_module_path,
    load_from_module,
    mkdir
)


Callback = tf.keras.callbacks.Callback


def load_keras_callbacks(callbacks: List[dict],
                         out_dir: Path) -> List[Type[Callback]]:
    """Returns a list of deserialized callbacks.

    Args:
    calbacks_config : List[dict]
        A list of serialized Keras callbacks.

    out_dir : Path
        Directory that will contain folder 'callbacks'
        with nested subfolders for each callback that
        produces outputs to disk (e.g., ModelCheckpoint,
        CSVLogger, TensorBoard)
    """

    result = [load_keras_callback(c) for c in callbacks]
    result = [c for c in result if c]

    # Adjust relative paths.
    # (Use hasattr in case custom subclasses are used)
    for c in result:
        c_cls = c.__class__.__name__
        # if isinstance(c, tf.keras.callbacks.ModelCheckpoint):
        if hasattr(c, 'filepath'):
            c.filepath = str(mkdir(out_dir / c_cls) / c.filepath)

        # if isinstance(c, tf.keras.callbacks.CSVLogger):
        elif hasattr(c, 'filename'):
            c.filename = str(mkdir(out_dir / c_cls) / c.filename)

        # if isinstance(c, tf.keras.callbacks.TensorBoard):
        elif hasattr(c, 'log_dir'):
            c.log_dir = str(mkdir(out_dir / c_cls / c.log_dir))

    return result


def load_keras_callback(identifier: dict) -> Type[Callback]:
    """Returns deserialized Keras based callback object.

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

    path = join_module_path(__name__, class_name) or f'tf.k.callbacks.{class_name}'
    return load_from_module(path, **config)


class MergedTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir=Path('./logs'), **kwargs):
        self.train_log_dir = log_dir / 'train'
        self.valid_log_dir = log_dir / 'valid'

        super().__init__(str(self.train_log_dir), **kwargs)

    def set_model(self, model):
        self.val_writer = tf.summary.FileWriter(str(self.valid_log_dir))
        super().set_model(model)

    def on_batch_end(self, batch, logs=None):
        super().on_batch_end(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        val_logs = {('epoch_' + k.replace('val_', '')): v
                    for k, v in logs.items()
                    if k.startswith('val_')}

        if self.update_freq == 'epoch':
            step = epoch
        else:
            step = self._samples_seen

        # TODO Implement v2 Summary when running eagerly -- lines 1099-1106 from TensorBoard callback
        for name, value in val_logs.items():
            if isinstance(value, np.ndarray):
                value - value.item()
            summary = tf.summary.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.val_writer.add_summary(summary, step)
        self.val_writer.flush()

        train_logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super().on_epoch_end(epoch, train_logs)

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        self.val_writer.close()
