from __future__ import division
from __future__ import print_function

from collections import OrderedDict, defaultdict, namedtuple
from cached_property import cached_property
import ipdb
import pdb
import itertools as it
import numpy as np
import os
import threading
import time

try:
  import cPickle as pickle  # py2
except ImportError:
  import pickle  # py3

import tensorflow as tf
import sonnet as snt


def _not_iterable(obj):
  return not snt.nest.is_iterable(obj)


def _not_iterable_or_dict(obj):
  return not snt.nest.is_iterable(obj) and not isinstance(obj, dict)


def _discard_trailing_nones(*args):
  args = list(args)
  while args[-1] is None:
    args.pop(-1)

  if len(args) == 0:
    return None
  elif len(args) == 1:
    return args[0]
  else:
    return tuple(args)


def shape_if_known(tensor, dim):
  val = tensor.get_shape()[dim].value
  if val is None:
    val = tf.shape(tensor)[dim]
  return val


def list_if_not(lst):
  if not snt.nest.is_sequence(lst):
    lst = [lst]
  return list(lst)


def concat_shapes(*shapes):
  shape = tf.TensorShape([])
  for s in shapes:
    shape = shape.concatenate(tf.TensorShape(s))
  return shape


def map_nested(func, nested, until=_not_iterable_or_dict):
  if until(nested):
    return func(nested)

  elif isinstance(nested, dict):
    output = Struct()
    for k, v in nested.items():
      output[k] = map_nested(func, v, until)
    return output

  elif snt.nest.is_iterable(nested):
    return tuple(map_nested(func, v, until) for v in nested)

  else:
    raise ValueError


def _list_flatten(structure):
  out = []
  if snt.nest.is_sequence(structure) and not isinstance(structure, dict):
    for s in structure:
      out.extend(_list_flatten(s))
  else:
    out = [structure]
  return out


def Session(devices=None, frac=None, monitored=False, **kwargs):
  """Create a session."""
  if devices is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = devices

  SessionType = tf.train.MonitoredSession if monitored else tf.Session

  if frac is not None:
    kwargs['gpu_options'] = tf.GPUOptions(per_process_gpu_memory_fraction=frac)

  return SessionType(config=tf.ConfigProto(allow_soft_placement=True, **kwargs))


def pl_def(*shape, dtype=tf.float32):
  return Struct(shape=concat_shapes(*shape), dtype=dtype)


def make_placeholders(variables):
  """Shortcut to make placholders.

  Args:
    variables: A dict where the keys are str and values are `tfu.PlDef` or nested list/tuple structures of them.

  Returns:
    A dict(name: tf.placeholder) with the same keys as `variables`.
  """
  placeholders = Struct()
  for name, args in variables.items():
    placeholders[name] = map_nested(lambda pl: tf.placeholder(pl.dtype, pl.shape, name),
                                    args, until=lambda x: isinstance(x, Struct))

  return placeholders


def pl_like_tensor(tensor, name=None):
  return tf.placeholder(tensor.dtype, tensor.get_shape().as_list(),
                        name=name)


def placeholders_like(tensors):
  return map_nested(pl_like_tensor, tensors)


class Struct(dict):
  """A dict that exposes its entries as attributes."""

  def __init__(self, **kwargs):
    dict.__init__(self, kwargs)
    self.__dict__ = self

  @staticmethod
  def make(obj):
    """Modify `obj` by replacing `dict`s with `tfu.Struct`s."""
    if isinstance(obj, dict):
      if isinstance(obj, Struct):
        ObjType = type(obj)
      else:
        ObjType = Struct
      return ObjType(**{k: Struct.make(v) for k, v in obj.items()})

    elif isinstance(obj, list):
      return [Struct.make(v) for v in obj]

    return obj


class Function(object):

  def __init__(self, inputs, outputs,
               session=tf.get_default_session, name='function'):
    """Create a function interface to `session.run()`.

    Args:
      inputs: a dict, keys are strings and values are nested
        Structures that can be a `feed_dict`. Leaves are placeholders,
        and will be replaced by numpy arrays when the function
        is evaluated.
      outputs: any (nested) Structure that can be evaluated
        by `session.run()`.
      session: a callable that returns a `tf.Session`.
      name: a string.
    """
    self.session, self.name = session, name
    self.inputs, self.outputs = inputs, outputs

  def __call__(self, **values):
    session = self.session()
    feed = {pl: values[name]
            for name, pl in self.inputs.items()}
    result = session.run(self.outputs, feed_dict=feed)
    return Struct.make(result)

  def __str__(self):
    return '< tfu.Function: %s >' % self.name

  def __repr__(self):
    return str(self)


class BaseModel(snt.AbstractModule):

  scope_name = 'Model'

  def __init__(self, hp):
    super(BaseModel, self).__init__(name=self.scope_name)
    self.hp = hp
    self.session = tf.get_default_session

  def _build(self):
    return self

  def snapshot(self, path, trainable_only=False):
    if trainable_only:
      variables = tf.trainable_variables()
    else:
      variables = tf.global_variables()

    variables = self.session().run({
        v.name: v for v in variables
        if v.name.startswith(self.scope_name)
    })

    snapshot = dict(
        variables=variables, hp=self.hp,
    )
    with open(path + '_vars.pkl', 'wb') as f:
      pickle.dump(snapshot, f, protocol=2)

  @classmethod
  def restore(cls, path, session, constructor_args):
    with open(path, 'rb') as f:
      snapshot = Struct.make(pickle.load(f))

    model = cls(snapshot.hp, *constructor_args)
    model.set_session(session)
    model.load_from(snapshot.variables)
    return model

  def load_from(self, path_or_dict):
    if isinstance(path_or_dict, str):
      with open(path_or_dict, 'rb') as f:
        values = pickle.load(f)
    else:
      values = path_or_dict

    var_list = self.variables

    assign_ops = []
    for v in var_list:
      name = str(v.name)
      if name not in values:
        print('WARNING: %s not in snapshot' % name)
        continue
      assign_ops.append(v.assign(values[name]))

    self.session().run(assign_ops)

  @property
  def variables(self):
    return [v for v in tf.global_variables()
            if v.name.startswith(self.scope_name)]

  @property
  def trainable_variables(self):
    return [v for v in tf.trainable_variables()
            if v.name.startswith(self.scope_name)]

  @property
  def num_params(self):
    return sum(np.prod(v.get_shape().as_list())
               for v in self.trainable_variables)


_optimizers = {
    'adam': tf.train.AdamOptimizer,
}


def _get_lr(global_step, args):
  lr = tf.constant(args.lr, dtype=tf.float32)
  decay_type = args.get('decay_type', None)

  if decay_type == 'exp':
    lr = tf.train.exponential_decay(
        lr, global_step, args.lr_step, args.lr_decay
    )

  elif decay_type != None:
    raise NotImplementedError(
        'lr decay type: %s' % args.decay_type
    )

  lr = tf.maximum(lr, args.get('min_lr', 0.))
  return lr


def adam(**overrides):
  default = Struct.make(dict(
      method='adam', lr=1e-3,
      args=dict(beta1=0.9, beta2=0.999, epsilon=1e-6),
  ))
  default.update(overrides)
  return default


def make_train_op(loss, opt, var_list=None,
                  update_ops_collection=tf.GraphKeys.UPDATE_OPS):

  step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)

  learning_rate = _get_lr(step, opt)
  optimizer = _optimizers[opt.method](learning_rate, **opt.args)

  if var_list is None:
    var_list = tf.trainable_variables()

  train_op = optimizer.minimize(loss, step)

  if update_ops_collection is not None:
    update_ops = tf.get_collection(
        update_ops_collection
    )
    train_op = tf.group(train_op, *update_ops)

  return Struct(
      it=step, lr=learning_rate, train_op=train_op, loss=loss,
  )


class PyfuncRunner(tf.train.queue_runner.QueueRunner):
  """Load data produced by arbitrary python code.

  Args:
    variables: A dict(str: tuple) describing the shapes returned by `func()`.
    capacity: Queue capacity, int.
    num_threads: Number of threads, int.
    gen_batches: Boolean, if true then the queue elements returned by `func()` are entire batches.
    func: Should return either a single training example or a batch of examples (depending on the values of `gen_batches`),
      in the format of a dict with the same keys as `variables` but with the values filled in as numpy arrays.
      Each runner thread will call `func()` independently, so it must be thread-safe.
    args, kwargs: Will be passed to `func()`.

  The runner threads can be paused and the queue can be flushed.

  If `gen_batches` is True, then the shapes in `variables` need not be fully-defined (as we don't need to call `dequeue_many()`).
  """

  def __init__(self, variables, capacity, num_threads,
               gen_batches, func, *args, **kwargs):
    self.gen_batches = gen_batches
    self.placeholders = make_placeholders(variables)

    all_shapes_defined = all(v.get_shape().is_fully_defined()
                             for v in self.placeholders.values())
    if all_shapes_defined:
      # If all shapes are fully-defined, construct the queue_accordingly.
      shapes = [pl.get_shape() for name, pl in self.placeholders.items()]
    else:
     #     assert not gen_batches, 'All shapes must be fully-defined if not queueing batches!'
      shapes = None

    queue = tf.FIFOQueue(capacity, shapes=shapes,
                         names=[name for name, pl in self.placeholders.items()],
                         dtypes=[pl.dtype for name, pl in self.placeholders.items()])

    enqueue_ops = [queue.enqueue(self.placeholders)] * num_threads

    self._session = tf.get_default_session()
    self._num_threads = num_threads
    self._func, self._args, self._kwargs = func, args, kwargs

    # Used to pause/resume runner threads.
    self._flag = threading.Event()
    self._flag.set()

    super(PyfuncRunner, self).__init__(queue, enqueue_ops)

    self.queue_size = Function({}, queue.size(), self.session)

    if gen_batches:
      self.get_batch = Function({}, self.batch(), self.session)

    else:
      batch_size = tf.placeholder(tf.int32, [])
      self.get_batch = Function(dict(batch_size=batch_size),
                                self.batch(batch_size), self.session)

  def reset(self, restart=False):
    self.pause()
    while self.queue_size() > 0:
      if self.gen_batches:
        self.get_batch()
      else:
        self.get_batch(batch_size=self.queue_size())
    if restart:
      self.resume()
    return self

  def pause(self):
    self._flag.clear()
    return self

  def resume(self):
    self._flag.set()
    return self

  def create_threads(self, sess, coord=None, daemon=False, start=False):
    self._session = sess
    return super(PyfuncRunner, self).create_threads(sess, coord, daemon, start)

  def session(self):
    return self._session

  @property
  def is_paused(self):
    return not self._flag.is_set()

  def _run(self, sess, enqueue_op, coord=None):
    """Thread main function.

    This is exactly the same as `tf.QueueRunner`, except we enqueue the values generated by `func()`.
    """
    my_id = threading.get_ident()
    decremented = False
    try:
      do_enqueue = Function(self.placeholders, enqueue_op,
                            session=lambda: sess)

      def enqueue_callable():
        batch = self._func(*self._args, **self._kwargs)
        if batch is not None:
          do_enqueue(**batch)

      prev = -1
      while True:
        if coord and coord.should_stop():
          break

        # Only check if should pause ~once per sec.
        if time.time() - prev > 1.0:
          self._flag.wait()
          prev = time.time()

        try:
          enqueue_callable()

        except self._queue_closed_exception_types:
          with self._lock:
            self._runs_per_session[sess] -= 1
            decremented = True
            if self._run_per_session[sess] == 0:
              try:
                sess.run(self._close_op)
              except Exception as e:
                print('Ignored exception: %s' % str(e))

    except Exception as e:
      if coord:
        coord.request_stop(e)
      else:
        print('Exception in QueueRunner: %s' % str(e))
        with self._lock:
          self._exceptions_raised.append(e)
        raise
    finally:
      if not decremented:
        with self._lock:
          self._runs_per_session[sess] -= 1

  def batch(self, batch_size=None):
    """Get a batch of tensors."""
    if self.gen_batches:
      assert batch_size is None, 'Cannot enforce a batch size if `func()` returns batches!'
      batch = self._queue.dequeue()
      for name, pl in self.placeholders.items():
        shape = pl.get_shape()
        if shape.ndims is not None:
          batch[name].set_shape(shape.as_list())

    else:
      batch = self._queue.dequeue_many(batch_size)

    return Struct.make(batch)


def global_init(sess):
  sess.run(tf.global_variables_initializer())


def start_queue_runners(sess):
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess, coord)
  coord._threads = threads
  return coord


def stop_queue_runners(coord):
  coord.request_stop()
  coord.join(coord._threads)
  del coord._threads


Feature = Struct(
    bytes=lambda v: tf.train.Feature(bytes_list=tf.train.BytesList(value=v)),
    float=lambda v: tf.train.Feature(float_list=tf.train.FloatList(value=v)),
    int=lambda v: tf.train.Feature(int64_list=tf.train.Int64List(value=v)),
)

FeatureKeys = Struct(
    float=lambda s: tf.FixedLenFeature(s, tf.float32),
    bytes=lambda s: tf.FixedLenFeature(s, tf.string),
)


def Example(**data):
  return tf.train.Example(
      features=tf.train.Features(feature=data)
  )


def parse_example(ser):
  ex = tf.train.Example()
  ex.ParseFromString(ser)
  return Struct(**{
      k: v for k, v in ex.features.feature.items()
  })


def sample_unique(arr, num):
  idx = tf.random_shuffle(tf.range(tf.shape(arr)[0]))[:num]
  idx.set_shape([num])
  return tf.gather(arr, idx)


def leaky_relu(x, leak=0.1, name='relu'):
  with tf.variable_scope(name):
    f1, f2 = 0.5 * (1 + leak), 0.5 * (1 - leak)
    return f1 * x + f2 * tf.abs(x)


def gumbel_sample(logits):
  noise = tf.random_uniform(tf.shape(logits))
  idx = tf.argmax(logits - tf.log(-tf.log(noise)), axis=-1)
  samples = tf.one_hot(idx, logits.get_shape()[-1].value)
  return samples


def flip_batch_time_major(tensor):
  return tf.transpose(tensor, (1, 0) + tuple(range(2, tensor.get_shape().ndims)))


def layer_factory(layer_defs, name_prefix='layer', list_flatten=True):
  layers = []
  layer_names = []
  if list_flatten:
    layer_defs = _list_flatten(layer_defs)
  for i, layer in enumerate(layer_defs):
    if isinstance(layer, list):
      sublayers, sublayer_names = layer_factory(
          layer, name_prefix='%s_%d' % (name_prefix, i), list_flatten=True)
      layers.append(sublayers)
      layer_names.append(sublayer_names)
    else:
      if isinstance(layer, dict):
        cls_name = layer.get('name')
        name = '%s_%s_%d' %  \
            (name_prefix, cls_name[cls_name.index('.') + 1:], i)
        cls_args = layer.get('args', [])
        cls_kwargs = layer.get('kwargs', {})
      else:
        cls_name, cls_args, cls_kwargs = layer, [], {}
        name = '%s_%d' % (name_prefix, i)
      cls_kwargs['name'] = name
      if cls_name.startswith('tfu.'):
        cls_name = cls_name[4:]
      cls = eval(cls_name)
      if isinstance(cls, type):
        layers.append(cls(*cls_args, **cls_kwargs))
        layer_names.append(cls_name)
      else:
        def outer_func():
          curr_cls_kwargs = dict(cls_kwargs)
          curr_cls_args = list(cls_args)
          curr_cls = eval(cls_name)

          def func(*x, **kwargs):
            args = list(x) + list(curr_cls_args)
            kwargs.update(curr_cls_kwargs)
            return curr_cls(*args, **kwargs)
          return func
        layers.append(outer_func())
        layer_names.append(cls_name)

  return layers, layer_names


class Pool2D(snt.AbstractModule):

  def __init__(self, ksize, strides=(1, 1),
               padding=snt.SAME, mode='max', name='Pool2D'):
    super(Pool2D, self).__init__(name=name)

    self._mode = mode
    self._params = dict(
        ksize=[1] + list(ksize) + [1],
        strides=[1] + list(strides) + [1],
        padding=padding,
    )

  def _build(self, inputs):
    if self._mode == 'max':
      pool_func = tf.nn.max_pool
    elif self._mode == 'mean':
      pool_func = tf.nn.avg_pool
    else:
      raise ValueError(self._mode)

    return pool_func(inputs, **self._params)


def _make_queued_step(core, inputs, dones):
  initial_state = snt.nest.flatten(
      core.initial_state(shape_if_known(dones, 0)))
  queue = tf.FIFOQueue(capacity=1, dtypes=[t.dtype for t in initial_state])

  def _do_reset():
    dequeued = queue.dequeue()
    for t, v in zip(dequeued, initial_state):
      t.set_shape(v.get_shape())

    new_hidden = [tf.where(dones, initial_val, current_val)
                  for initial_val, current_val in zip(initial_state, dequeued)]
    return queue.enqueue(new_hidden)

  reset_op = tf.cond(tf.equal(queue.size(), 0),
                     lambda: queue.enqueue(initial_state),
                     _do_reset)

  dequeued = queue.dequeue()
  for t, v in zip(dequeued, initial_state):
    t.set_shape(v.get_shape())

  outputs, next_hidden = core(inputs,
                              snt.nest.pack_sequence_as(core.state_size, dequeued))
  step_op = queue.enqueue(snt.nest.flatten(next_hidden))
  return outputs, step_op, reset_op


class StackedRNN(snt.AbstractModule):

  _supports_conv_mode = set()

  def __init__(self, cores, name=None):
    super(StackedRNN, self).__init__(name=name)
    self._cores = cores
    can_use_conv = all(type(c) in self._supports_conv_mode for c in cores
                       if isinstance(c, snt.AbstractModule))
    with self._enter_variable_scope():
      self._conv_core = snt.Sequential(cores) if can_use_conv else None
      self._rnn_core = snt.DeepRNN(
          cores, skip_connections=False) if cores else None

  def _build(self, inputs, prev_hidden=None):
    if inputs.get_shape().ndims == 3:
      assert prev_hidden is None
      next_hidden = None
      if len(self._cores) == 0:
        outputs = inputs
      elif self._conv_core is not None:
        outputs = self._conv_core(inputs)
      else:
        outputs, _ = tf.nn.dynamic_rnn(
            self._rnn_core, inputs, dtype=tf.float32)

    elif inputs.get_shape().ndims == 2:
      if len(self._cores) == 0:
        outputs, next_hidden = inputs, prev_hidden
      else:
        assert prev_hidden is not None
        outputs, next_hidden = self._rnn_core(inputs, prev_hidden)

    else:
      raise ValueError

    return _discard_trailing_nones(outputs, next_hidden)

  @property
  def state_size(self):
    return self._rnn_core.state_size if self._rnn_core else ()

  def initial_state(self, *args, **kwargs):
    return self._rnn_core.initial_state(*args, **kwargs) if self._rnn_core else ()

  @classmethod
  def can_use_conv(cls, other_cls):
    cls._supports_conv_mode.add(other_cls)
    return other_cls

  def make_queued_step(self, inputs_pl, dones_pl):
    outputs = inputs_pl
    step_ops, reset_ops = [], []
    for core in self._cores:
      if hasattr(core, 'make_queued_step'):
        outputs, step_op, reset_op = core.make_queued_step(outputs, dones_pl)
      elif isinstance(core, snt.RNNCore):
        outputs, step_op, reset_op = _make_queued_step(core, outputs, dones_pl)
      else:
        outputs = core(outputs)
        step_op = reset_op = tf.no_op(), tf.no_op()
      step_ops.append(step_op), reset_ops.append(reset_op)
    return outputs, step_ops, reset_ops


class Identity(snt.AbstractModule):

  def __init__(self, name='Identity'):
    super(Identity, self).__init__(name=name)

  def _build(self, inputs):
    return inputs


@StackedRNN.can_use_conv
class Conv1x1(snt.AbstractModule):

  def __init__(self, output_size, use_bias=True, name='Conv1x1'):
    super(Conv1x1, self).__init__(name=name)
    with self._enter_variable_scope():
      self._core = snt.Linear(output_size=output_size, use_bias=use_bias)

  def _build(self, inputs):
    while isinstance(inputs, tuple):
      # snt.Sequential passing (obs, hidden_state) as inputs, where hidden_state is None
      inputs = inputs[0]

    ndims = inputs.get_shape().ndims
    if ndims == 2:
      return self._core(inputs)
    else:
      return snt.BatchApply(self._core, ndims - 1)(inputs)


@StackedRNN.can_use_conv
class CausalConv1D(snt.RNNCore):

  def __init__(self, params, train_initial_state=True,
               residual=False, dense=False,
               name=None):
    super(CausalConv1D, self).__init__(name=name)
    self._params = Struct.make(params)
    self._train_initial_state = train_initial_state
    self._input_channels = None

    if residual and dense:
      raise ValueError('Cannot have residual and dense connections!')
    if residual not in {False, 'up', 'down'}:
      raise ValueError('residual should one of [False, "up", "down"]')
    self._residual, self._dense = residual, dense

    with self._enter_variable_scope():
      self._cores = Struct(
          conv_f=snt.Conv1D(name='conv_xf', padding=snt.VALID, **params),
          conv_g=snt.Conv1D(name='conv_xg', padding=snt.VALID, **params),
          lin_f=snt.Linear(self._params.output_channels, name='lin_zf'),
          lin_g=snt.Linear(self._params.output_channels, name='lin_zg'),
      )

  def _build(self, x_in, h_in=None, z=None):
    """
    inputs = (x, z) (conv)
    inputs = (x, h) (rnn)
    inputs = (x, z, h) (rnn)
    """
    if isinstance(x_in, tuple):
      # snt.Sequential passing (obs, hidden_state) as x_in, where hidden_state is None
      x_in = x_in[0]

    mode = None
    if x_in.get_shape().ndims == 3:
      mode = 'conv'

    elif x_in.get_shape().ndims == 2:
      mode = 'rnn'
      assert h_in is not None

    if mode is None:
      raise ValueError('Could not determine mode from inputs with shapes: %s' %
                       ', '.join(x.get_shape() for x in inputs))

    if self.is_connected:
      if self._cores.lin_f.is_connected and z is None:
        raise ValueError('Z was given earlier but is missing now.')
      if not self._cores.lin_f.is_connected and z is not None:
        raise ValueError('Z is given now but was not earlier.')

    k, rate = self._params.kernel_shape, self._params.rate
    pad_size = rate * (k - 1)
    c_in = x_in.get_shape()[-1].value
    if self._input_channels is None:
      self._input_channels = c_in
    elif self._input_channels != c_in:
      raise ValueError(
          'Expected %d input channels but got %d!' % (
              self._input_channels, c_in))

    if mode == 'conv':
      # conv mode
      if h_in is None:
        if self._train_initial_state:
          padding = self.initial_state(tf.shape(x_in)[0])
          xx = tf.concat(list(padding) + [x_in], axis=1)
        else:
          xx = tf.pad(x_in, [(0, 0), (pad_size, 0), (0, 0)])
      else:
        raise NotImplementedError

      xf, xg = self._cores.conv_f(xx), self._cores.conv_g(xx)
      h_out = None

    elif mode == 'rnn':
      # rnn mode
      assert k == 2, "only K=2 is supported for RNN"
      _x_in = tf.expand_dims(x_in, axis=1)
      h_out = snt.nest.flatten(list(h_in) + [_x_in])
      h_out_len = len(h_out)
      h_out = tf.concat(h_out, axis=1)
      xx = tf.stack([h_out[:, 0], h_out[:, -1]], axis=1)
      #h_out = tf.concat(h_out[1:], axis=1)
      h_out = (h_out[:, 1:],)

      if h_out_len == pad_size + 1:
        xf, xg = self._cores.conv_f(xx)[:, -1], self._cores.conv_g(xx)[:, -1]
      else:
        xf = tf.einsum('bki,kij->bj', xx, self._cores.conv_f.w)
        xg = tf.einsum('bki,kij->bj', xx, self._cores.conv_g.w)
        if self._cores.conv_f.has_bias:
          xf += self._cores.conv_f.b
        if self._cores.conv_g.has_bias:
          xg += self._cores.conv_g.b

      # h_out = tf.split(h_out[:, 1:], k - 1, axis=1)

    if z is not None:
      xf += self._cores.lin_f(z)
      xg += self._cores.lin_g(z)

    x_out = tf.tanh(xf) * tf.sigmoid(xg)

    if self._dense:
      x_out = tf.concat([x_in, x_out], axis=-1)
    elif self._residual == 'up':
      if self._params.output_channels == x_in.get_shape()[-1]:
        x_out += x_in
      else:
        x_out += Conv1x1(self._params.output_channels, name='res_up')(x_in)
    elif self._residual == 'down':
      if self._params.output_channels == x_in.get_shape()[-1]:
        x_out += x_in
      else:
        x_out = Conv1x1(c_in, name='res_down')(x_out) + x_in

    return _discard_trailing_nones(x_out, h_out, z)

  def make_queued_step(self, inputs, dones):
    k, rate = self._params.kernel_shape, self._params.rate
    queues = [tf.FIFOQueue(rate, dtypes=tf.float32) for _ in range(k - 1)]
    counter_queue = tf.FIFOQueue(1, dtypes=tf.int32)

    def _new_counters(batch_size):
      return tf.ones([batch_size], tf.int32) * rate

    def _partial_reset_counters():
      cur_counters = counter_queue.dequeue()
      cur_counters.set_shape([None])
      initial_counters = _new_counters(tf.shape(cur_counters)[0])
      new_counters = tf.where(dones, initial_counters, cur_counters)
      return counter_queue.enqueue(new_counters)

    def _full_reset_counters():
      with tf.control_dependencies([_flush_queue(counter_queue)]):
        return counter_queue.enqueue(_new_counters(tf.shape(dones)[0]))

    reset_op = tf.cond(tf.equal(counter_queue.size(), 0) | tf.reduce_all(dones),
                       _full_reset_counters, _partial_reset_counters)

    counters = counter_queue.dequeue()
    counters.set_shape([None])
    batch_size = shape_if_known(inputs, 0)
    fill_value = tf.tile(self._initial_state_var, [batch_size, 1])
    xx, x_to_push, push_ops = [], inputs, []

    for i in range(k - 1):
      x_popped = _cond_dequeue(queues[i],
                               tf.equal(queues[i].size(), rate),
                               fill_value)
      x_popped = tf.where(tf.greater(counters, 0), fill_value, x_popped)

      push_ops.append(queues[i].enqueue([x_to_push]))
      x_to_push = x_popped
      xx.append(x_popped)

    h_in = list(reversed(xx))
    outputs, _ = self(inputs, xx)

    push_ops.append(counter_queue.enqueue(tf.maximum(counters - 1, 0)))
    return outputs, push_ops, reset_op

  @property
  def output_size(self):
    return self._params.output_channels

  @cached_property
  def state_size(self):
    assert self._input_channels is not None
    k, rate = self._params.kernel_shape, self._params.rate
    return tuple(tf.TensorShape([rate, self._input_channels]) for _ in range(k - 1))

  @cached_property
  def _initial_state_var(self):
    if self._train_initial_state:
      return snt.TrainableVariable([1, self._input_channels],
                                   initializers=dict(w=tf.zeros_initializer),
                                   name='initial_state')()
    else:
      return tf.zeros([1, self._input_channels])

  def initial_state(self, batch_size, dtype=tf.float32, **unused_kwargs):
    trainable = self._train_initial_state
    k, rate = self._params.kernel_shape, self._params.rate

    if dtype == tf.resource:
      return tuple(tf.FIFOQueue(rate, dtypes=tf.float32) for _ in range(k - 1))

    elif dtype.is_integer:
      return tf.ones([batch_size], dtype=dtype) * rate

    elif trainable:
      state = tf.tile(self._initial_state_var[:, None], [batch_size, rate, 1])
      return tuple(state for _ in range(k - 1))

    else:
      return super(CausalConv1D, self).initial_state(batch_size, dtype=tf.float32, trainable=False)


def _cond_dequeue(queue, switch_bool, fill_value):
  dequeued = tf.cond(switch_bool, queue.dequeue, lambda: fill_value)
  dequeued.set_shape(fill_value.get_shape())
  return dequeued


def _flush_queue(queue):
  def cond(size):
    return tf.greater(size, 0)

  def body(size):
    with tf.control_dependencies([queue.dequeue()]):
      return queue.size()

  return tf.while_loop(cond, body, [queue.size()], back_prop=False)


@StackedRNN.can_use_conv
class DenseBlock(snt.RNNCore):

  def __init__(self, params, name='DenseBlock'):
    super(DenseBlock, self).__init__(name=name)
    with self._enter_variable_scope():
      self.layers, _ = layer_factory(params)
      self._cores = Struct(
          out=StackedRNN(self.layers)
      )

  def _build(self, obs, prev_hidden=None):
    if obs.get_shape().ndims == 3:
      assert prev_hidden is None
      ip = self._cores.out(obs)
      next_hidden = None

    elif obs.get_shape().ndims == 2:
      assert prev_hidden is not None

      if isinstance(prev_hidden, tf.Tensor) and prev_hidden.dtype == tf.bool:
        step_ops, reset_ops = [], []
        dones = prev_hidden
        ip, step_ops, reset_ops = self._cores.out.make_queued_step(obs, dones)
        next_hidden = step_ops, reset_ops
      else:
        ip, next_hidden = self._cores.out(obs, prev_hidden)

    else:
      raise ValueError

    output = tf.concat([obs, ip], axis=-1)
    return _discard_trailing_nones(output, next_hidden)

  @property
  def output_size(self):
    return self.layers[-1].output_size

  def initial_state(self, batch_size, dtype=tf.float32, **unused):
    return self._cores.out.initial_state(batch_size, dtype, **unused),

  @property
  def state_size(self):
    return tuple(snt.nest.flatten(self._cores.out.state_size))


class ResBlock(snt.AbstractModule):

  def __init__(self, params, shrink_factor=1, name='ResBlock'):
    super(ResBlock, self).__init__(name=name)
    self.shrink_factor = shrink_factor
    with self._enter_variable_scope():
      self.layers, self.layer_names = layer_factory(params)
      self._cores = Struct(
          out=snt.Module(self._custom_build),
          preprocess=snt.BatchNorm(decay_rate=0.99, scale=True, fused=True)
      )

  def _build(self, obs, is_training=True):
    # assuming square spatial dimensions
    preact = self._cores.preprocess(obs, is_training=is_training)
    preact = leaky_relu(preact)

    ip = self._cores.out(preact, is_training)

    out_channels = ip.get_shape().as_list()[-1]
    in_channels = obs.get_shape().as_list()[-1]
    if in_channels != out_channels:
      shortcut = preact
    else:
      shortcut = obs

    if self.shrink_factor != 1:
      area = (self.shrink_factor, self.shrink_factor)
      shortcut = Pool2D(mode='max',
                        ksize=[1, 1],
                        strides=area)(shortcut)

    if in_channels != out_channels:
      shortcut = Conv1x1(out_channels)(shortcut)

    return shortcut + ip

  def _custom_build(self, obs, is_training=True):

    def _call_with_args(layer, name, ip):
      if 'BatchNorm' in name:
        return layer(ip, is_training)
      elif 'dropout' in name and not is_training:
        return ip
      else:
        return layer(ip)

    out = obs
    for layer, name in zip(self.layers, self.layer_names):
      out = _call_with_args(layer, name, out)

    return out


@StackedRNN.can_use_conv
class CausalAttention(snt.RNNCore):

  def __init__(
          self, buffer_size, keys_dim, vals_dim,
          query_dim=None, num_heads=1, last_timestep_only=False,
          postprocess=None, dense=True, attend_over_self=False, name=None):
    """A Recurrent module that performs soft attention over the input. At
    timestep t, the output o = softmax(K^Tq/sqrt(d))V, where 'q' is the
    embedded input at timestep 't', K = [k_1, ... k_{t-1}] is a matrix of
    the previous inputs embedded as keys. V = [v_1, ... , v_{t-1}] is the same,
    but for the values

    Args:
      buffer_size: How far back in the temporal dimension
        the module will attend over
      keys_dim: The embedding dimension size for the keys
      vals_dim: The embedding dimension size for the values
      query_dim: The embedding dimension size for the queries
      num_heads: Number of heads of attention
      last_timestep_only: When the input is an entire sequence, if true, this module
        will only provide an output for the last timestep
      postprocess: additional processing on the attention output before the read
        vector is concatenated with the input
      dense: if true, concat the input with the output read vector
      atend_over_self: if true, also attends over current timestep
      name: Scope name for parameters in this module

    Returns:
      Let B be the batch size, d' and d the channel dimensions of the input and output
      respectively. Outputs (B, d + d') or (B, d) matrix if the input is (B, d'),
      (B, T, d+d') or (B, T, d+d') if the input is (B, T, d') and 'last_timestep_only'
      is false; otherwise output is of the form (B,d+d'). the output channels are
      d+d' if dense is true and d otherwise
    """
    super(CausalAttention, self).__init__(name=name)
    with self._enter_variable_scope():
      if isinstance(keys_dim, int):
        keys = Conv1x1(keys_dim * num_heads, name='key_module')
      if isinstance(vals_dim, int):
        vals = Conv1x1(vals_dim * num_heads, name='vals_module')
      if isinstance(query_dim, int):
        query = Conv1x1(query_dim * num_heads, name='query_module')
      if postprocess:
        postprocess = snt.Sequential(layer_factory(
            postprocess, name_prefix='output_module')[0])
    if keys_dim is None:
      keys = Identity()
    if query_dim is None:
      query = keys

    self._cores = Struct(keys=keys, vals=vals, query=query,
                         postprocess=postprocess)
    self._keys_dim = keys_dim
    self._vals_dim = vals_dim
    self._buffer_size = buffer_size
    self._num_heads = num_heads
    self._dense = dense
    self._last_timestep_only = last_timestep_only
    self._attend_over_self = attend_over_self

  def _build(self, inputs, prev_hidden=None):
    if isinstance(inputs, tuple):
      # snt.Sequential passing (obs, hidden_state) as inputs, where hidden_state is None
      inputs = inputs[0]
    batch_size = shape_if_known(inputs, 0)
    if self._keys_dim == None:
      self._keys_dim = shape_if_known(inputs, -1)

    if inputs.get_shape().ndims == 3:
      # pdb.set_trace()
      key_seq = self._cores.keys(inputs)
      val_seq = self._cores.vals(inputs)

      d, D = self._keys_dim, self._vals_dim
      t = shape_if_known(key_seq, 1)

      key_seq = tf.reshape(key_seq, [batch_size, t, self._num_heads, d])
      val_seq = tf.reshape(val_seq, [batch_size, t, self._num_heads, D])

      if self._last_timestep_only:
        query = self._cores.query(inputs[:, -1])
        query = tf.reshape(query, [batch_size, self._num_heads, d])

        if not self._attend_over_self:
          # do not want to attend over current timestep
          key_seq = key_seq[:, :-1]
          val_seq = val_seq[:, :-1]

        logits = tf.einsum('bkhd,bhd->bkh', key_seq, query) / np.sqrt(d)

        probs = tf.nn.softmax(logits, dim=1)
        tf.add_to_collection('pr', probs)

        read = tf.einsum('bkh,bkhd->bhd', probs, val_seq)
        read = tf.reshape(read, [batch_size, self._num_heads * self._vals_dim])
      else:
        # Do the entire sequence of attention in one pass.
        # We compute a [T, T] matrix of `logits`, and mask out a triangular section to make it causal.
        # Then we softmax over the last dimension to a [T] vector of `probs`.
        # Note that if you compare `probs` or `logits` in conv vs rnn mode, they will differ by a permutation.
        # However, because of how rnn-mode updates its hidden state, the keys/values will also differ by the same permutation,
        # so the results of the attentive read will be the same. When num_heads > 1, we repeat this process a num_heads
        # number of times, and concatenate the respective outputs

        if self._attend_over_self:
          def cond(i, j): return j <= i
        else:
          def cond(i, j): return j < i

        def get_idx(t):
          return np.array(sorted([(0, j, 0, i) for j in range(t)
                                  for i in range(t) if i - self._buffer_size <= j and cond(i, j)]))

        query_seq = self._cores.query(inputs)
        query_seq = tf.reshape(query_seq, [batch_size, t, self._num_heads, d])

        logits = tf.einsum('bkhd,bqhd->bkhq', key_seq, query_seq) / np.sqrt(d)

        # Use a py_func to compute mask if sequence length is not known yet.
        if isinstance(t, tf.Tensor):
          indices = tf.py_func(get_idx, [t], tf.int64, stateful=False)
          indices.set_shape([None, 4])
          shape = tf.to_int64(tf.stack([1, t, 1, t], axis=0))

        else:
          indices = tf.constant(get_idx(t), tf.int64)
          shape = [1, t, 1, t]

        idx_sparse = tf.SparseTensor(
            indices,
            tf.ones([shape_if_known(indices, 0)], tf.bool),
            shape,
        )
        mask = tf.tile(tf.sparse_tensor_to_dense(idx_sparse, False),
                       [shape_if_known(logits, 0), 1, self._num_heads, 1])
        logits = tf.where(mask, logits, -32 * tf.ones_like(logits))
        probs = tf.nn.softmax(logits, dim=1)
        read = tf.einsum('bkhq,bkhd->bhqd', probs, val_seq)

        if not self._attend_over_self:
          # A slight weirdness but needed for correctness.
          read = tf.concat([tf.zeros([batch_size,  self._num_heads, 1, self._vals_dim]),
                            read[:, :,  1:]], axis=2)
        # transposing to (B,T,H,d) so we can reshape to (B,T,d*H)
        read = tf.transpose(read, [0, 2, 1, 3])
        read = tf.reshape(
            read, [batch_size, t, self._num_heads * self._vals_dim])

      next_hidden = None

    elif inputs.get_shape().ndims == 2:
      assert prev_hidden is not None
      next_hidden = []
      key_seq, val_seq, mask_seq = prev_hidden

      d, D = self._keys_dim, self._vals_dim
      query = self._cores.query(inputs)
      key_seq = tf.reshape(
          key_seq, [batch_size, self._buffer_size, self._num_heads, d])
      val_seq = tf.reshape(
          val_seq, [batch_size, self._buffer_size, self._num_heads, D])
      query = tf.reshape(query, [batch_size, self._num_heads, d])

      logits = tf.einsum('bkhd,bhd->bkh', key_seq, query) / np.sqrt(d)
      logits = tf.where(mask_seq, logits, -32 * tf.ones_like(logits))
      probs = tf.nn.softmax(logits, dim=1)
      tf.add_to_collection('pr', probs)

      read = tf.einsum('bkh,bkhd->bhd', probs, val_seq)

      inputs_expanded = tf.expand_dims(inputs, 1)
      curr_keys = tf.reshape(self._cores.keys(inputs_expanded), [
          batch_size, 1, self._num_heads, d])
      curr_vals = tf.reshape(self._cores.vals(inputs_expanded), [
          batch_size, 1, self._num_heads, D])
      key_seq = tf.concat([curr_keys, key_seq[:, :-1]], axis=1)
      val_seq = tf.concat([curr_vals, val_seq[:, :-1]], axis=1)
      mask_seq = tf.concat(
          [tf.ones([batch_size, 1, self._num_heads], tf.bool), mask_seq[:, :-1]], axis=1)
      next_hidden = key_seq, val_seq, mask_seq
      read = tf.reshape(read, [batch_size, self._num_heads * self._vals_dim])

    else:
      raise ValueError

    if self._cores.postprocess:
      read = self._cores.postprocess(read)

    if self._dense:
      if self._last_timestep_only and inputs.get_shape().ndims == 3:
        output = tf.concat([inputs[:, -1], read], axis=-1)
      else:
        output = tf.concat([inputs, read], axis=-1)
    else:
      output = read
    return _discard_trailing_nones(output, next_hidden)

  @property
  def output_size(self):
    return self._vals_dim * self._num_heads

  @property
  def state_size(self):
    return (
        tf.TensorShape([self._buffer_size, self._num_heads, self._keys_dim]),
        tf.TensorShape([self._buffer_size, self._num_heads, self._vals_dim]),
        tf.TensorShape([self._buffer_size, self._num_heads]),
    )

  def initial_state(self, batch_size, **unused_kwargs):
    assert self._keys_dim and self._vals_dim
    shape = [batch_size, self._buffer_size, self._num_heads]

    if self._attend_over_self:
      mask_seq = tf.concat([
          tf.ones([batch_size, 1, self._num_heads], dtype=tf.bool),
          tf.zeros([batch_size, self._buffer_size -
                    1, self._num_heads], dtype=tf.bool)
      ], axis=1)
    else:
      mask_seq = tf.zeros(
          [batch_size, self._buffer_size, self._num_heads], tf.bool)

    return (
        tf.zeros(shape + [self._keys_dim]),
        tf.zeros(shape + [self._vals_dim]),
        mask_seq
    )


"""


def discounted_cumsum(values, discount, out=None):
  if np.isscalar(discount):
    discount *= np.ones_like(values)
  assert values.shape == discount.shape
  if out is None:
    out = np.zeros_like(values)
  assert out.shape == values.shape

  T = values.shape[1]
  for t in reversed(range(T)):
    out[:, t] = values[:, t]
    if t + 1 < T:
      out[:, t] += discount[:, t] * out[:, t + 1]

  return out


def compute_advantages(rewards, values, gamma, lam, out=None):
  assert rewards.shape == values.shape
  if np.isscalar(gamma):
    gamma *= np.ones_like(values)
  assert gamma.shape == rewards.shape

  T = rewards.shape[1]
  deltas = np.zeros_like(rewards)
  for t in reversed(range(T)):
    deltas[:, t] = rewards[:, t] + gamma[:, t] * \
        (values[:, t + 1] if t + 1 < T else 0) - values[:, t]

  return discounted_cumsum(deltas, gamma * lam, out)

"""
