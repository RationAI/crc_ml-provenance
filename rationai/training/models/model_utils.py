import tensorflow as tf  # get rid of this
from tensorflow.python.autograph.lang import directives
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine.training import concat
from tensorflow.python.keras.engine.training import reduce_per_replica
from tensorflow.python.keras.engine.training import _keras_api_gauge
from tensorflow.python.keras.engine.training import _minimum_control_deps
from tensorflow.python.keras.engine.training import _disallow_inside_tf_function
# from tensorflow.python.keras.utils import tf_inspect # not in v2.3.1
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.profiler import trace
from tensorflow.python.util import nest
# https://stackoverflow.com/questions/58441514/why-is-tensorflow-2-much-slower-than-tensorflow-1
from typing import Dict
from typing import Tuple

# Aliases for type hints
Metric = tf.keras.metrics.Metric
Tensor = tf.python.framework.ops.Tensor


def test_step(self, data: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Dict[str, Metric]]:
    """Overrides Keras Model's test_step method.
    Both predictions and metrics are returned."""
    data = data_adapter.expand_1d(data)
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

    y_pred = self(x, training=False)
    # Updates stateful loss metrics.
    self.compiled_loss(
        y, y_pred, sample_weight, regularization_losses=self.losses)

    self.compiled_metrics.update_state(y, y_pred, sample_weight)
    return y_pred, {m.name: m.result() for m in self.metrics}


def make_test_function(self):
    if self.test_function is not None:
        return self.test_function

    def step_function(model, iterator):
        """Runs a single evaluation step.
        Returns both prediction and metrics."""

        def run_step(data):
            outputs = model.test_step(data)
            # Ensure counter is updated only if `test_step` succeeds.
            with ops.control_dependencies(_minimum_control_deps(outputs)):
                model._test_counter.assign_add(1)  # pylint: disable=protected-access
            return outputs

        data = next(iterator)
        y_pred, metrics = model.distribute_strategy.run(run_step, args=(data,))
        y_pred = reduce_per_replica(
            y_pred, self.distribute_strategy, reduction='first')
        metrics = reduce_per_replica(
            metrics, self.distribute_strategy, reduction='first')
        return y_pred, metrics

    if self._steps_per_execution.numpy().item() == 1:
        def test_function(iterator):
            """Runs an evaluation execution with one step."""
            return step_function(self, iterator)
    else:
        def test_function(iterator):
            """Runs an evaluation execution with multiple steps."""
            y_pred, metrics = step_function(self, iterator)
            for _ in math_ops.range(self._steps_per_execution - 1):
                directives.set_loop_options(
                    shape_invariants=[(
                        t, tf_utils.get_tensor_spec(t, dynamic_batch=True).shape)
                        for t in nest.flatten(y_pred)])
                step_y_pred, metrics = step_function(self, iterator)
                y_pred = nest.map_structure(lambda t1, t2: concat([t1, t2]), y_pred,
                                            step_y_pred)
            return y_pred, metrics

    if not self.run_eagerly:
        test_function = def_function.function(
            test_function, experimental_relax_shapes=True)

    self.test_function = test_function
    return self.test_function


def evaluate(self,
             x=None,
             y=None,
             batch_size=None,
             verbose=1,
             sample_weight=None,
             steps=None,
             callbacks=None,
             max_queue_size=10,
             workers=1,
             use_multiprocessing=False,
             return_dict=False):
    """Generates output predictions for the input samples
    and computes the loss value & metrics values for the model in test mode.
    """
    _keras_api_gauge.get_cell('evaluate').set(True)
    version_utils.disallow_legacy_graph('Model', 'evaluate')
    self._assert_compile_was_called()
    self._check_call_args('evaluate')
    _disallow_inside_tf_function('evaluate')

    y_preds = None
    with self.distribute_strategy.scope():
        data_handler = data_adapter.DataHandler(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps,
            initial_epoch=0,
            epochs=1,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            model=self,
            steps_per_execution=self._steps_per_execution)

        # Container that configures and calls `tf.keras.Callback`s.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self,
                verbose=verbose,
                epochs=1,
                steps=data_handler.inferred_steps)

        batch_y_preds = None  # model predictions
        logs = {}  # metric logs
        self.test_function = self.make_test_function()
        self._test_counter.assign(0)
        callbacks.on_predict_begin()
        callbacks.on_test_begin()

        for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
            self.reset_metrics()
            with data_handler.catch_stop_iteration():
                for step in data_handler.steps():
                    with trace.Trace('test', step_num=step, _r=1):
                        callbacks.on_predict_batch_begin(step)
                        callbacks.on_test_batch_begin(step)
                        tmp_batch_y_preds, tmp_logs = self.test_function(iterator)
                        if data_handler.should_sync:
                            context.async_wait()
                        logs = tmp_logs  # No error, now safe to assign to logs.
                        batch_y_preds = tmp_batch_y_preds  # No error, now safe to assign to logs.

                        if y_preds is None:
                            y_preds = nest.map_structure(
                                lambda batch_y_pred: [batch_y_pred], batch_y_preds)
                        else:
                            nest.map_structure_up_to(
                                batch_y_preds,
                                lambda output, batch_y_pred: output.append(batch_y_pred),
                                y_preds, batch_y_preds)

                        end_step = step + data_handler.step_increment
                        callbacks.on_predict_batch_end(end_step, {'outputs': batch_y_preds})
                        callbacks.on_test_batch_end(end_step, logs)
        if batch_y_preds is None:
            raise ValueError('Expect x to be a non-empty array or dataset.')

        all_y_preds = tf_utils.to_numpy_or_python_type(
            nest.map_structure_up_to(batch_y_preds, concat, y_preds))
        logs = tf_utils.to_numpy_or_python_type(logs)

        # Two calls together cause 2nd metric progress bar
        # to appear at the end of model.evaluate()
        callbacks.on_predict_end()
        callbacks.on_test_end(logs=logs)

        if return_dict:
            return {'logs': logs, 'y_preds': all_y_preds}
        else:
            results = []
            for name in self.metrics_names:
                if name in logs:
                    results.append(logs[name])
            for key in sorted(logs.keys()):
                if key not in self.metrics_names:
                    results.append(logs[key])
            if len(results) == 1:
                return results[0]
            return results, all_y_preds
