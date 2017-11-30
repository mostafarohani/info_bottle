import tensorflow as tf
import numpy as np
import sonnet as snt
import tf_utils as tfu
import argparse
import os
import os.path as osp
import pickle
import tcml_util as util
import ipdb


class Generator(snt.AbstractModule):
  def __init__(self, name, dim_in, dim_out):
    super(Generator, self).__init__(name=name)

    with self._enter_variable_scope():
      layer_defs = [
          util.attention(keys_dim=128, vals_dim=32,
                         query_dim=128, max_path_length=dim_in),
          'tfu.leaky_relu',
          util.tc_block(32, [1, 2, 4, 8, 16]),
          util.attention(keys_dim=128, vals_dim=32,
                         query_dim=128, max_path_length=dim_in),
          'tfu.leaky_relu',
          util.tc_block(32, [1, 2, 4, 8, 16]),
          util.attention(keys_dim=128, vals_dim=32,
                         query_dim=128, max_path_length=dim_in),
          'tfu.leaky_relu',
          util.conv1x1(dim_out),
      ]
      self.layers, self.layer_names = tfu.layer_factory(layer_defs)
      self._core = snt.Sequential(self.layers)

  def _build(self, obs):
    out = self._core(obs)
    return out


def train_generator(name, train_data, test_data, bins=32):
  lb, ub = test_data.min(), test_data.max()
  train_data = np.clip(train_data, lb, ub)
  dataset = tf.data.Dataset.from_tensor_slices(train_data)
  dataset = dataset.shuffle(512).batch(32).prefetch(32).repeat()
  iterator = dataset.make_initializable_iterator()
  tf.add_to_collection('datasets', iterator.initializer)

  model = Generator(name, train_data.shape[-1], bins)

  x_raw = iterator.get_next()
  x_discr = tf.py_func(lambda x: np.digitize(x, np.linspace(lb, ub, bins)),
                       [x_raw], tf.int64)
  x_discr.set_shape(x_raw.shape)
  x = tf.one_hot(x_discr, bins)

  x_shifted = tf.concat([tf.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
  logits = model(x_shifted)

  ds = tf.contrib.distributions
  nll = tf.negative(tf.reduce_mean(ds.Categorical(logits).log_prob(x_discr)))

  train = tfu.make_train_op(nll, tfu.adam())
  train.stats = tfu.Struct(nll=nll)

  return train,


if __name__ == '__main__':
  all_iters = [0, 5, 10, 20, 1000, 9000]
  all_layers = [1, 3, 4]

  data = dict()
  for itr in all_iters:
    file_loc = osp.join('data', '%d.pkl' % itr)
    with open(file_loc, 'rb') as f:
      data.update(pickle.load(f))

  train_ops = []
  for itr in all_iters:
    for layer in all_layers:
      prefix = 'itr%d_layer%d' % (itr, layer)
      train_data = data['train_%s' % prefix]
      test_data = data['test_%s' % prefix]
      train_ops.append(train_generator(prefix, train_data, test_data))

  train = tfu.Function({}, train_ops)

  sess = tfu.Session(devices='0').__enter__()
  tfu.global_init(sess)
  sess.run(tf.get_collection('datasets'))
