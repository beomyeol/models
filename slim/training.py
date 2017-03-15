import time
import os

import tensorflow as tf

from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import coordinator
from tensorflow.python.training import optimizer as tf_optimizer
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import supervisor
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.training import training_util
from tensorflow.python.training import basic_session_run_hooks as tf_hooks
from tensorflow.python.training import summary_io

metrics = {
  # 'total_train_time': 0.0,
  'total_checkpoint_time': 0.0,
  'checkpoint_count': 0.0
}

def add_metrics(key, value):
  if not key in metrics:
    raise ValueError('Unknown key: ' + key)
  metrics[key] += value

def get_metrics(key):
  return metrics[key]

_USE_DEFAULT = 0

def train(train_op,
          logdir,
          logging_tensors=None,
          log_every_n_steps=1,
          graph=None,
          master='',
          is_chief=True,
          global_step=None,
          number_of_steps=None,
          init_op=_USE_DEFAULT,
          init_feed_dict=None,
          local_init_op=_USE_DEFAULT,
          init_fn=None,
          ready_op=_USE_DEFAULT,
          summary_op=_USE_DEFAULT,
          save_summaries_secs=600,
          summary_writer=_USE_DEFAULT,
          startup_delay_steps=0, # not used..
          checkpoint_basename='model.ckpt',
          async_checkpoint=False,
          saver=None,
          save_steps=None,
          save_secs=600,
          sync_optimizer=None,
          session_config=None,
          trace_every_n_steps=None):
  """Runs a training loop using a TensorFlow supervisor.

  When the sync_optimizer is supplied, gradient updates are applied
  synchronously. Otherwise, gradient updates are applied asynchronous.

  Args:
    train_op: A `Tensor` that, when executed, will apply the gradients and
      return the loss value.
    logdir: The directory where training logs are written to. If None, model
      checkpoints and summaries will not be written.
    train_step_fn: The function to call in order to execute a single gradient
      step. The function must have take exactly four arguments: the current
      session, the `train_op` `Tensor`, a global step `Tensor` and a dictionary.
    train_step_kwargs: A dictionary which is passed to the `train_step_fn`. By
      default, two `Boolean`, scalar ops called "should_stop" and "should_log"
      are provided.
    log_every_n_steps: The frequency, in terms of global steps, that the loss
      and global step and logged.
    graph: The graph to pass to the supervisor. If no graph is supplied the
      default graph is used.
    master: The address of the tensorflow master.
    is_chief: Specifies whether or not the training is being run by the primary
      replica during replica training.
    global_step: The `Tensor` representing the global step. If left as `None`,
      then slim.variables.get_or_create_global_step() is used.
    number_of_steps: The max number of gradient steps to take during training.
      If the value is left as None, training proceeds indefinitely.
    init_op: The initialization operation. If left to its default value, then
      the session is initialized by calling `tf.global_variables_initializer()`.
    init_feed_dict: A feed dictionary to use when executing the `init_op`.
    local_init_op: The local initialization operation. If left to its default
      value, then the session is initialized by calling
      `tf.local_variables_initializer()` and `tf.tables_initializer()`.
    init_fn: An optional callable to be executed after `init_op` is called. The
      callable must accept one argument, the session being initialized.
    ready_op: Operation to check if the model is ready to use. If left to its
      default value, then the session checks for readiness by calling
      `tf.report_uninitialized_variables()`.
    summary_op: The summary operation.
    save_summaries_secs: How often, in seconds, to save summaries.
    summary_writer: `SummaryWriter` to use.  Can be `None`
      to indicate that no summaries should be written. If unset, we
      create a SummaryWriter.
    startup_delay_steps: The number of steps to wait for before beginning. Note
      that this must be 0 if a sync_optimizer is supplied.
    saver: Saver to save checkpoints. If None, a default one will be created
      and used.
    save_secs: How often, in seconds, to save the model to `logdir`.
    sync_optimizer: an instance of tf.train.SyncReplicasOptimizer. If the
      argument is supplied, gradient updates will be synchronous. If left as
      `None`, gradient updates will be asynchronous.
    session_config: An instance of `tf.ConfigProto` that will be used to
      configure the `Session`. If left as `None`, the default will be used.
    trace_every_n_steps: produce and save a `Timeline` in Chrome trace format
      and add it to the summaries every `trace_every_n_steps`. If None, no trace
      information will be produced or saved.

  Returns:
    the value of the loss function after training.

  Raises:
    ValueError: if `train_op` is empty or if `startup_delay_steps` is
      non-zero when `sync_optimizer` is supplied, if `number_of_steps` is
      negative, or if `trace_every_n_steps` is not `None` and no `logdir` is
      provided.
  """
  if train_op is None:
    raise ValueError('train_op cannot be None.')

  if logdir is None:
    if summary_op != _USE_DEFAULT:
      raise ValueError('Cannot provide summary_op because logdir=None')
    if trace_every_n_steps is not None:
      raise ValueError('Cannot provide trace_every_n_steps because '
                       'logdir=None')

  if sync_optimizer is not None and startup_delay_steps > 0:
    raise ValueError(
        'startup_delay_steps must be zero when sync_optimizer is supplied.')

  if number_of_steps is not None and number_of_steps <= 0:
    raise ValueError(
        '`number_of_steps` must be either None or a positive number.')

  if save_steps is not None and save_secs is not None:
    raise ValueError('Either `save_secs` or `save_steps` must be set, not both.')

  if log_every_n_steps and log_every_n_steps > 0:
    if not logging_tensors:
      raise ValueError('Cannot provide log_every_n_steps because '
                       'logging_tensors=None')

  graph = graph or ops.get_default_graph()
  with graph.as_default():
    if global_step is None:
      global_step = variables.get_or_create_global_step()

    # checkpoint ops
    saver = saver if saver else tf_saver.Saver()
    save_path = None if not logdir else os.path.join(logdir, checkpoint_basename)

    with ops.name_scope('init_ops'):
      if init_op == _USE_DEFAULT:
        init_op = tf_variables.global_variables_initializer()

      if ready_op == _USE_DEFAULT:
        ready_op = tf_variables.report_uninitialized_variables()

      if local_init_op == _USE_DEFAULT:
        local_init_op = control_flow_ops.group(
            tf_variables.local_variables_initializer(),
            data_flow_ops.tables_initializer())

      if sync_optimizer is not None and isinstance(
          sync_optimizer, sync_replicas_optimizer.SyncReplicasOptimizer):
        with ops.control_dependencies([local_init_op] if local_init_op is
                                      not None else []):
          if is_chief:
            local_init_op = sync_optimizer.chief_init_op
          else:
            local_init_op = sync_optimizer.local_step_init_op
        ready_for_local_init_op = sync_optimizer.ready_for_local_init_op
      else:
        ready_for_local_init_op = None

    if summary_op == _USE_DEFAULT:
      summary_op = summary.merge_all()

    if is_chief and sync_optimizer is not None:
      if not isinstance(sync_optimizer,
                        (sync_replicas_optimizer.SyncReplicasOptimizer)):
        raise ValueError(
            '`sync_optimizer` must be a tf.train.SyncReplicasOptimizer.')

      # Need to create these BEFORE the supervisor finalizes the graph:
      init_tokens_op = sync_optimizer.get_init_tokens_op()
      chief_queue_runner = sync_optimizer.get_chief_queue_runner()

  scaffold = tf.train.Scaffold(init_op=init_op,
                               init_feed_dict=init_feed_dict,
                               init_fn=init_fn,
                               ready_op=ready_op,
                               ready_for_local_init_op=ready_for_local_init_op,
                               local_init_op=local_init_op,
                               summary_op=summary_op,
                               saver=saver)

  hooks = [tf.train.StopAtStepHook(last_step=number_of_steps)]

  if summary_writer == _USE_DEFAULT:
    summary_writer = summary_io.SummaryWriterCache.get(logdir)

  # summary
  if save_summaries_secs and save_summaries_secs > 0:
    hooks.append(tf.train.SummarySaverHook(save_steps=None,
                                           save_secs=save_summaries_secs,
                                           output_dir=logdir,
                                           summary_writer=summary_writer,
                                           scaffold=scaffold))

  # logging
  if log_every_n_steps and log_every_n_steps > 0:
    hooks.append(tf.train.LoggingTensorHook(logging_tensors,
                                            every_n_iter=log_every_n_steps,
                                            every_n_secs=None))
  # checkpoint
  if not async_checkpoint:
    if (save_secs and save_secs > 0) or (save_steps and save_steps > 0):
      listeners = [TimerCheckpointSaverListener()]
      hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=logdir,
                                                save_secs=save_secs,
                                                save_steps=save_steps,
                                                scaffold=scaffold,
                                                checkpoint_basename=checkpoint_basename,
                                                listeners=listeners))
  else:
    if (not save_secs or save_secs == 0) and (save_steps and save_steps > 0):
      raise ValueError('Cannot perform step-based asynchronous checkpoint.')

  with tf.train.MonitoredTrainingSession(master=master,
                                         is_chief=is_chief,
                                         checkpoint_dir=None, # TODO: can be restored??
                                         scaffold=scaffold,
                                         hooks=hooks,
                                         chief_only_hooks=None,
                                         save_checkpoint_secs=None, # disable default
                                         save_summaries_steps=None, # disable default
                                         config=session_config) as mon_sess:
    logging.info('Starting Session.')
    checkpoint_thread = None
    if async_checkpoint and save_secs and save_secs > 0:
      checkpoint_thread = CheckpointThread(mon_sess._coordinated_creator.coord,
                                           global_step, summary_writer,
                                           mon_sess._tf_sess(),
                                           saver, save_path, save_secs).start()
    try:
      while not mon_sess.should_stop():
        start_time = time.time()
        # mon_sess.run handles AbortedError in case of preempted PS.
        total_loss, np_global_step = mon_sess.run([train_op, global_step])
        elapsed_time = time.time() - start_time

      logging.info('Stopping Training.')

    except:
      logging.info('An exception is thrown.')
      if is_chief and cleanup_op is not None:
        logging.info('About to execute sync_clean_up_op!')
        mon_sess.run(cleanup_op)
      raise

  return total_loss

class TimerCheckpointSaverListener(tf_hooks.CheckpointSaverListener):

  def __init__(self):
    pass

  def begin(self):
    pass

  def before_save(self, sess, global_step_value):
    self._start_time = time.time()

  def after_save(self, sess, global_step_value):
    elapsed_time = time.time() - self._start_time
    add_metrics('total_checkpoint_time', elapsed_time)
    add_metrics('checkpoint_count', 1)
    logging.info('%d-th checkpoint. time: %.2f sec',
        get_metrics('checkpoint_count'), elapsed_time)

  def end(self, sess, global_step_value):
    tf.logging.info('total checkpoint time: %.2f sec, # of checkpoints: %d',
          get_metrics('total_checkpoint_time'), get_metrics('checkpoint_count'))
    pass


class CheckpointThread(coordinator.LooperThread):

  def __init__(self, coord, global_step, summary_writer,
               sess, saver, save_path, save_secs):
    super(CheckpointThread, self).__init__(coord, save_secs)
    self._global_step = global_step
    self._summary_writer = summary_writer
    self._saver = saver
    self._sess = sess
    self._save_path = save_path
    # logging.info('Checkpoint thread is initialized: per %d secs to %s',
    #     save_secs, save_path)

  def run_loop(self):
    start_time = time.time()
    self._saver.save(self._sess, self._save_path, self._global_step)
    time_elapsed = time.time() - start_time
    add_metrics('checkpoint_count', 1)
    add_metrics('total_checkpoint_time', time_elapsed)

    current_step = training_util.global_step(self._sess, self._global_step)
    logging.info('Saving checkpoint. step: %d', current_step)
    if self._summary_writer and self._global_step is not None:
      self._summary_writer.add_session_log(
          SessionLog(status=SessionLog.CHECKPOINT,
                     checkpoint_path=self._save_path),
          current_step)

  def stop_loop(self):
    tf.logging.info('total checkpoint time: %.2f sec, # of checkpoints: %d',
          get_metrics('total_checkpoint_time'), get_metrics('checkpoint_count'))