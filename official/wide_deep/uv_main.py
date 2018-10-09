import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf

from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.wide_deep import uv_dataset
from official.wide_deep import wide_deep_run_loop


def define_uv_flags():
  wide_deep_run_loop.define_wide_deep_flags()
  flags.adopt_module_key_flags(wide_deep_run_loop)
  flags_core.set_defaults(data_dir='uv_dataset',
                          model_dir='uv_model',
                          train_epochs=1,
                          epochs_between_evals=1,
                          inter_op_parallelism_threads=0,
                          intra_op_parallelism_threads=0,
                          batch_size=500)


def build_estimator(model_dir, model_type, model_column_fn, inter_op, intra_op):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = model_column_fn()
  hidden_units = [100, 50]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0},
                                    inter_op_parallelism_threads=inter_op,
                                    intra_op_parallelism_threads=intra_op))

  if model_type == 'wide':
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)
  elif model_type == 'deep':
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)
  else:
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config)


def run_uv(flags_obj):
  """Construct all necessary functions and call run_loop.

  Args:
    flags_obj: Object containing user specified flags.
  """

  train_file = os.path.join(flags_obj.data_dir, uv_dataset.TRAINING_FILE)
  test_file = os.path.join(flags_obj.data_dir, uv_dataset.EVAL_FILE)

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
  def train_input_fn():
    return uv_dataset.input_fn(
        train_file, flags_obj.epochs_between_evals, True, flags_obj.batch_size)

  def eval_input_fn():
    return uv_dataset.input_fn(test_file, 1, False, flags_obj.batch_size)

  tensors_to_log = {
      'average_loss': '{loss_prefix}head/truediv',
      'loss': '{loss_prefix}head/weighted_loss/Sum'
  }

  wide_deep_run_loop.run_loop(
      name="uv level experiments", train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      model_column_fn=uv_dataset.build_model_columns,
      build_estimator_fn=build_estimator,
      flags_obj=flags_obj,
      tensors_to_log=tensors_to_log,
      early_stop=True)


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_uv(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_uv_flags()
  absl_app.run(main)
