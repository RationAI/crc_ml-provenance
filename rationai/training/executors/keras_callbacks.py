import tensorflow as tf
import logging
from datetime import datetime

prov_log = logging.getLogger('prov-training.callback')

class ProvenanceCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        prov_log.info(f'Train Start: {datetime.now()}')

    def on_train_end(self, logs=None):
        prov_log.info(f'Train End: {datetime.now()}')

    def on_epoch_begin(self, epoch, logs=None):
        prov_log.info(f'Train Epoch {epoch} Start: {datetime.now()}')

    def on_epoch_end(self, epoch, logs=None):
        prov_log.info(f'Train Epoch {epoch} End: {datetime.now()}')
        for key in logs:
            prov_log.info(f'Epoch {epoch} {key}: {logs[key]:.2f}')

    