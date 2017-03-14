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

def default_train_step_kwargs(global_step,
                              log_every_n_steps,
                              number_of_steps):
  train_step_kwargs = {}

  with ops.name_scope('train_step'):
    if number_of_steps:
      should_stop_op = math_ops.greater_equal(global_step, number_of_steps)
    else:
      should_stop_op = constant_op.constant(False)
    train_step_kwargs['should_stop'] = should_stop_op
    train_step_kwargs['should_log'] = math_ops.equal(
        math_ops.mod(global_step, log_every_n_steps), 0)

    train_time = tf.placeholder(tf.float32, name='train_time')
    train_step_kwargs['train_time'] = train_time
    train_step_kwargs['total_train_time'] = state_ops.assign_add(
        tf.Variable(name='total_train_time', initial_value=0.0,
                    trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]),
        train_time)

    # if is_chief and trace_every_n_steps is not None:
    #   train_step_kwargs['should_trace'] = math_ops.equal(
    #       math_ops.mod(global_step, trace_every_n_steps), 0)
    #   train_step_kwargs['logdir'] = logdir

  return train_step_kwargs

def train_step(sess, train_op, global_step, train_step_kwargs):
  """Function that takes a gradient step and specifies whether to stop.

  Args:
    sess: The current session.
    train_op: An `Operation` that evaluates the gradients and returns the
      total loss.
    global_step: A `Tensor` representing the global training step.
    train_step_kwargs: A dictionary of keyword arguments.

  Returns:
    The total loss and a boolean indicating whether or not to stop training.

  Raises:
    Va lueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
  """
  start_time = time.time()

  trace_run_options = None
  run_metadata = None
  if 'should_trace' in train_step_kwargs:
    if 'logdir' not in train_step_kwargs:
      raise ValueError('logdir must be present in train_step_kwargs when '
                       'should_trace is present')
    if sess.run(train_step_kwargs['should_trace']):
      trace_run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()

  total_loss, np_global_step = sess.run([train_op, global_step],
                                        options=trace_run_options,
                                        run_metadata=run_metadata)
  time_elapsed = time.time() - start_time

  if run_metadata is not None:
    tl = timeline.Timeline(run_metadata.step_stats)
    trace = tl.generate_chrome_trace_format()
    trace_filename = os.path.join(train_step_kwargs['logdir'],
                                  'tf_trace-%d.json' % np_global_step)
    logging.info('Writing trace to %s', trace_filename)
    file_io.write_string_to_file(trace_filename, trace)
    if 'summary_writer' in train_step_kwargs:
      train_step_kwargs['summary_writer'].add_run_metadata(run_metadata,
                                                           'run_metadata-%d' %
                                                           np_global_step)

  if 'should_log' in train_step_kwargs:
    if sess.run(train_step_kwargs['should_log']):
      logging.info('global step %d: loss = %.4f (%.2f sec/step)',
                   np_global_step, total_loss, time_elapsed)

  # TODO(nsilberman): figure out why we can't put this into sess.run. The
  # issue right now is that the stop check depends on the global step. The
  # increment of global step often happens via the train op, which used
  # created using optimizer.apply_gradients.
  #
  # Since running `train_op` causes the global step to be incremented, one
  # would expected that using a control dependency would allow the
  # should_stop check to be run in the same session.run call:
  #
  #   with ops.control_dependencies([train_op]):
  #     should_stop_op = ...
  #
  # However, this actually seems not to work on certain platforms.
  if 'should_stop' in train_step_kwargs:
    should_stop = sess.run(train_step_kwargs['should_stop'])
  else:
    should_stop = False

  # Gather total elapsed time
  total_train_time = sess.run(train_step_kwargs['total_train_time'],
                              feed_dict={train_step_kwargs['train_time']: time_elapsed})

  # checkpoint periodically
  if 'should_save' in train_step_kwargs:
    should_save = sess.run(train_step_kwargs['should_save'])
    if should_save:
      logging.info("Saving checkpoint. step: %d", np_global_step)
      checkpoint(sess, global_step,
                 saver=train_step_kwargs['saver'],
                 save_path=train_step_kwargs['save_path'],
                 save_counter_inc_op=train_step_kwargs['save_counter_inc_op'],
                 checkpoint_time=train_step_kwargs['checkpoint_time'],
                 checkpoint_time_add_op=train_step_kwargs['checkpoint_time_add_op'])
  else:
    should_save = False

  # log total elapsed time for training
  if should_stop:
    logging.info('total training time: %.2f sec, step: %d',
                 total_train_time, np_global_step)

  return total_loss, should_stop

def checkpoint(sess, global_step, saver, save_path,
               save_counter_inc_op, checkpoint_time, checkpoint_time_add_op):
  start_time = time.time()

  saver.save(sess, save_path, global_step)

  time_elapsed = time.time() - start_time

  sess.run([save_counter_inc_op, checkpoint_time_add_op],
           feed_dict={checkpoint_time: time_elapsed})

_USE_DEFAULT = 0

def train(train_op,
          logdir,
          train_step_fn=train_step,
          train_step_kwargs=_USE_DEFAULT,
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
          startup_delay_steps=0,
          checkpoint_basename='model.ckpt',
          saver=None,
          save_steps=0,
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

  if save_steps and save_secs:
    raise ValueError('Either `save_secs` or `save_steps` must be set, not both.')

  graph = graph or ops.get_default_graph()
  with graph.as_default():
    if global_step is None:
      global_step = variables.get_or_create_global_step()

    if train_step_kwargs == _USE_DEFAULT:
      train_step_kwargs = default_train_step_kwargs(global_step=global_step,
                                                    log_every_n_steps=log_every_n_steps,
                                                    number_of_steps=number_of_steps)

    # checkpoint ops
    saver = saver if saver else tf_saver.Saver()
    save_path = None if not logdir else os.path.join(logdir, checkpoint_basename)
    save_counter = tf.Variable(
        name='save_counter', initial_value=0,
        trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    save_counter_inc_op = state_ops.assign_add(save_counter, 1)
    checkpoint_time = tf.placeholder(tf.float32, name='checkpoint_time')
    total_checkpoint_time = tf.Variable(name='total_checkpoint_time', initial_value=0.0,
        trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    checkpoint_time_add_op = state_ops.assign_add(total_checkpoint_time, checkpoint_time)

    if save_steps:
      train_step_kwargs['saver'] = saver
      train_step_kwargs['save_path'] = save_path
      train_step_kwargs['should_save'] = math_ops.equal(
          math_ops.mod(global_step, save_steps), 0)
      train_step_kwargs['save_counter'] = save_counter
      train_step_kwargs['save_counter_inc_op'] = save_counter_inc_op
      train_step_kwargs['checkpoint_time'] = checkpoint_time
      train_step_kwargs['total_checkpoint_time'] = total_checkpoint_time
      train_step_kwargs['checkpoint_time_add_op'] = checkpoint_time_add_op

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

    if summary_writer == _USE_DEFAULT:
      summary_writer = supervisor.Supervisor.USE_DEFAULT

    cleanup_op = None

    if is_chief and sync_optimizer is not None:
      if not isinstance(sync_optimizer,
                        (sync_replicas_optimizer.SyncReplicasOptimizer)):
        raise ValueError(
            '`sync_optimizer` must be a tf.train.SyncReplicasOptimizer.')

      # Need to create these BEFORE the supervisor finalizes the graph:
      init_tokens_op = sync_optimizer.get_init_tokens_op()
      chief_queue_runner = sync_optimizer.get_chief_queue_runner()
      if isinstance(sync_optimizer,
                    sync_replicas_optimizer.SyncReplicasOptimizer):
        cleanup_op = sync_optimizer.get_clean_up_op()

  sv = supervisor.Supervisor(
      graph=graph,
      is_chief=is_chief,
      logdir=logdir,
      init_op=init_op,
      init_feed_dict=init_feed_dict,
      local_init_op=local_init_op,
      ready_for_local_init_op=ready_for_local_init_op,
      ready_op=ready_op,
      summary_op=summary_op,
      summary_writer=summary_writer,
      global_step=global_step,
      save_summaries_secs=save_summaries_secs,
      saver=None, # disable checkpoint service in supervisor
      save_model_secs=0, # disable checkpoint service in supervisor
      init_fn=init_fn)

  if summary_writer is not None:
    train_step_kwargs['summary_writer'] = sv.summary_writer

  should_retry = True
  while should_retry:
    try:
      should_retry = False
      with sv.managed_session(
          master, start_standard_services=False, config=session_config) as sess:
        logging.info('Starting Session.')
        if is_chief:
          if logdir:
            sv.start_standard_services(sess)
            # run checkpoint thread
            if save_secs:
              CheckpointThread(sv, sess, saver, save_path, save_secs,
                               save_counter_inc_op,
                               checkpoint_time,
                               checkpoint_time_add_op).start()
        elif startup_delay_steps > 0:
          _wait_for_step(sess, global_step,
                         min(startup_delay_steps, number_of_steps or
                             sys.maxint))
        sv.start_queue_runners(sess)
        logging.info('Starting Queues.')
        if is_chief and sync_optimizer is not None:
          sv.start_queue_runners(sess, [chief_queue_runner])
          sess.run(init_tokens_op)
        try:
          while not sv.should_stop():
            total_loss, should_stop = train_step_fn(sess, train_op, global_step,
                                                    train_step_kwargs)
            if should_stop:
              logging.info('Stopping Training.')
              break
          if logdir and sv.is_chief:
            logging.info('Finished training! Saving model to disk.')
            checkpoint(sess, sv.global_step, saver, save_path,
                       save_counter_inc_op, checkpoint_time, checkpoint_time_add_op)

            # logging checkpoint metrics
            np_total_checkpoint_time, np_save_counter = sess.run(
                [total_checkpoint_time, save_counter])
            tf.logging.info('total checkpoint time: %.2f sec, # of checkpoints: %d',
                np_total_checkpoint_time, np_save_counter)
        except:
          if sv.is_chief and cleanup_op is not None:
            logging.info('About to execute sync_clean_up_op!')
            sess.run(cleanup_op)
          raise

    except errors.AbortedError:
      # Always re-run on AbortedError as it indicates a restart of one of the
      # distributed tensorflow servers.
      logging.info('Retrying training!')
      should_retry = True

  return total_loss

class CheckpointThread(coordinator.LooperThread):

  def __init__(self, sv, sess, saver, save_path, save_secs,
               save_counter_inc_op, checkpoint_time, checkpoint_time_add_op):
    super(CheckpointThread, self).__init__(sv.coord, save_secs)
    self._sv = sv
    self._saver = saver
    self._sess = sess
    self._save_path = save_path
    self._save_counter_inc_op = save_counter_inc_op
    self._checkpoint_time = checkpoint_time
    self._checkpoint_time_add_op = checkpoint_time_add_op
    # logging.info('Checkpoint thread is initialized: per %d secs to %s',
    #     save_secs, save_path)

  def run_loop(self):
    checkpoint(self._sess, self._sv.global_step, self._saver, self._save_path,
               self._save_counter_inc_op, self._checkpoint_time, self._checkpoint_time_add_op)
    current_step = training_util.global_step(self._sess, self._sv.global_step)
    logging.info('Saving checkpoint. step: %d', current_step)
    if self._sv.summary_writer and self._sv.global_step is not None:
      self._sv.summary_writer.add_session_log(
          SessionLog(status=SessionLog.CHECKPOINT,
                     checkpoint_path=self._save_path),
          current_step)