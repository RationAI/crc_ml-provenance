import tensorflow as tf
from keras.utils import tf_utils
from keras.utils import io_utils
import logging

from rationai.utils.provenance import SummaryWriter

sw_log = SummaryWriter.getLogger('provenance')

class ProvenanceCallback(tf.keras.callbacks.Callback):
    def __init__(self, update_prov_epoch=True):
        super().__init__()
        self.update_epoch = update_prov_epoch

    def on_train_begin(self, logs=None):
        sw_log.set('train', 'start', value=SummaryWriter.now())

    def on_train_end(self, logs=None):
        sw_log.set('train', 'end', value=SummaryWriter.now())

    def on_predict_begin(self, logs=None):
        sw_log.set('predict', 'start', value=SummaryWriter.now())

    def on_predict_end(self, logs=None):
        sw_log.set('predict', 'end', value=SummaryWriter.now())

    def on_evaluate_begin(self, logs=None):
        sw_log.set('eval', 'start', value=SummaryWriter.now())

    def on_evaluate_end(self, logs=None):
        sw_log.set('eval', 'end', value=SummaryWriter.now())

    def on_epoch_begin(self, epoch, logs=None):
        sw_log.set('iters', sw_log.vars['gen_counter'], 'start', value=SummaryWriter.now())
        if self.update_epoch:
            sw_log.vars['gen_counter'] = epoch + 1

    def on_epoch_end(self, epoch, logs=None):
        sw_log.set('iters', sw_log.vars['gen_counter'], 'end', value=SummaryWriter.now())

class ProvenanceModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.last_save = None
        self.save_id = 0
        while self.save_id <= sw_log.vars['save_id']:
            self.save_id += 1
        sw_log.vars['save_id'] += 1

    def _save_model(self, epoch, batch, logs):
        """Saves the model. 
        
        Copied and modified original code from Keras.
        Added provenance logging.

        Args:
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
            is set to `epoch`.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        if isinstance(self.save_freq, int) or self.epochs_since_last_save >= self.period:
            # Block only when saving interval is reached.
            logs = tf_utils.sync_to_numpy_or_python_type(logs)
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, batch, logs)

            try:
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        logging.warning('Can save best model only with %s available, '
                                'skipping.', self.monitor)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                io_utils.print_msg(
                                    f'\nEpoch {epoch + 1}: {self.monitor} improved '
                                    f'from {self.best:.5f} to {current:.5f}, '
                                    f'saving model to {filepath}')
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(
                                    filepath, overwrite=True, options=self._options)
                            else:
                                self.model.save(filepath, overwrite=True, options=self._options)
                            if self.last_save is not None:
                                sw_log.set('iters', self.last_save, 'checkpoints', f'save_{self.save_id}', 'valid', value=False)
                            sw_log.set('iters', epoch, 'checkpoints', f'save_{self.save_id}', 'filepath', value=str(filepath))
                            sw_log.set('iters', epoch, 'checkpoints', f'save_{self.save_id}', 'valid', value=True)
                            self.last_save = epoch
                        else:
                            if self.verbose > 0:
                                io_utils.print_msg(
                                    f'\nEpoch {epoch + 1}: '
                                    f'{self.monitor} did not improve from {self.best:.5f}')
                else:
                    if self.verbose > 0:
                        io_utils.print_msg(
                            f'\nEpoch {epoch + 1}: saving model to {filepath}')
                    if self.save_weights_only:
                        self.model.save_weights(
                            filepath, overwrite=True, options=self._options)
                    else:
                        self.model.save(filepath, overwrite=True, options=self._options)

                    if self.last_save is not None and sw_log.get('iters', self.last_save, 'checkpoints', f'save_{self.save_id}', 'filepath') == str(filepath):
                        sw_log.set('iters', self.last_save, 'checkpoints', f'save_{self.save_id}', 'valid', value=False)
                    sw_log.set('iters', epoch, 'checkpoints', f'save_{self.save_id}', 'filepath', value=str(filepath))
                    sw_log.set('iters', epoch, 'checkpoints', f'save_{self.save_id}', 'valid', value=True)
                    self.last_save = epoch

                self._maybe_remove_file()
            except IsADirectoryError as e:  # h5py 3.x
                raise IOError('Please specify a non-directory filepath for '
                            'ModelCheckpoint. Filepath used is an existing '
                            f'directory: {filepath}')
            except IOError as e:  # h5py 2.x
                # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
                if 'is a directory' in str(e.args[0]).lower():
                    raise IOError('Please specify a non-directory filepath for '
                            'ModelCheckpoint. Filepath used is an existing '
                            f'directory: f{filepath}')
                # Re-throw the error for any other causes.
                raise e