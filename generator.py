import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import sonnet as snt
import tf_utils as tfu
import argparse
import os
import os.path as osp
import pickle
import tcml_util as util
import ipdb
from mpi4py import MPI


class Generator(snt.AbstractModule):
  def __init__(self, name, dim_in, dim_out):
    super(Generator, self).__init__(name=name)

    with self._enter_variable_scope():
      layer_defs = [
          util.tc_block(32, [1, 2, 4, 8, 16]),
          util.attention(keys_dim=32, vals_dim=32,
                         query_dim=32, max_path_length=dim_in),
          'tfu.leaky_relu',
          util.conv1x1(dim_out),
      ]
      self.layers, self.layer_names = tfu.layer_factory(layer_defs)
      self._core = snt.Sequential(self.layers)

  def _build(self, obs):
    out = self._core(obs)
    return out


def train_generator(name, data, bins=32):

  ds = tf.contrib.distributions

  def _build_model(model, cond_model, batch_size=32, split='train'):
    dataset = tf.data.Dataset.from_tensor_slices(dict(x=data['%s_%s' % (split, name)],
                                                      y=data['%s_labels' % split]))
    dataset = dataset.shuffle(512) \
                     .batch(batch_size) \
                     .prefetch(batch_size) \
                     .repeat()
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection('datasets', iterator.initializer)

    batch = tfu.Struct.make(iterator.get_next())

    # Discretize the activations into the given number of bins.
    x_discr = tf.py_func(lambda x: np.digitize(x, np.linspace(-1, 1, bins - 1)),
                         [batch.x], tf.int64)
    x_discr.set_shape(batch.x.shape)
    x = tf.one_hot(x_discr, bins)
    y = tf.one_hot(batch.y, 10)
    xy = tf.concat([x, tf.tile(y[:, None], [1, x.shape[1].value, 1])], axis=-1)

    # Estimate log p(X), H(X) = E[-log p(X)]
    x_shifted = tf.concat([tf.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
    xy_shifted = tf.concat([tf.zeros_like(xy[:, :1]), xy[:, :-1]], axis=1)
    logits = model(x_shifted)
    logits_cond = cond_model(xy_shifted)

    log_prob = ds.Categorical(logits).log_prob(x_discr)
    log_prob_cond = ds.Categorical(logits_cond).log_prob(x_discr)

    nll = tf.negative(tf.reduce_mean(log_prob))
    nll_cond = tf.negative(tf.reduce_mean(log_prob_cond))

    loss = nll + nll_cond

    # Estimate I(X;Y) = H(X) - H(X|Y)
    # H(X|Y) = E[-log p(X|Y=y) p(Y=y)]
    mi = nll
    for i in range(10):
      mask = tf.equal(batch.y, i)
      py = tf.reduce_mean(tf.to_float(mask))

      mi -= py * tf.cond(
          py > 0,
          lambda: tf.negative(tf.reduce_mean(
              tf.boolean_mask(log_prob_cond, mask))),
          lambda: tf.constant(0.0))

    ent = nll / np.log(2.)
    mi = mi / np.log(2.)

    return tfu.Struct(loss=loss,
                      ent=tf.check_numerics(ent, 'ENT is NaN!'),
                      mi=tf.check_numerics(mi, 'MI is NaN!'))

  model = Generator(name, train_data.shape[-1], bins)
  cond_model = Generator(name + '_cond', train_data.shape[-1] + 10, bins)
  train = _build_model(model, cond_model, split='train')
  train.op = tfu.make_train_op(train.loss, tfu.adam())
  test = _build_model(model, cond_model, split='test', batch_size=512)
  return train, test


if __name__ == '__main__':
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='tanh_data')
  parser.add_argument('--itr_count', type=int, default=10000)
  parser.add_argument('--print_window', type=int, default=100)
  parser.add_argument('--devices', type=str, default='0')
  FLAGS = parser.parse_args()

  sess = tfu.Session(devices=str(rank)).__enter__()
  all_iters = [0, 5, 10, 20, 1000, 9000]
  all_layers = [1, 3, 4]

  mnist = input_data.read_data_sets(FLAGS.data_dir)
  data = dict(train_labels=mnist.train.labels, test_labels=mnist.test.labels)
  for itr in all_iters:
    file_loc = osp.join(FLAGS.data_dir, '%d.pkl' % itr)
    with open(file_loc, 'rb') as f:
      data.update(pickle.load(f))

  outputs = {}
  count = 0
  for itr in all_iters:
    for layer in all_layers:
      if count % size == rank:
        prefix = 'itr%d_layer%d' % (itr, layer)
        train_data = data['train_%s' % prefix]
        test_data = data['test_%s' % prefix]
        print('creating model for %s' % prefix)
        train_ops, test_rollout = train_generator(prefix, data)
        train = tfu.Function({}, train_ops)
        test = tfu.Function({}, test_rollout)

        tfu.global_init(sess)
        sess.run(tf.get_collection('datasets'))
        print('starting training', prefix)

        for i in range(FLAGS.itr_count):
          tr = train()
          if i % FLAGS.print_window == 0:
            te = test()
            print('itr %d | train %.3f, %.3f | test %.3f, %.3f' %
                  (i, tr.ent, tr.mi, te.ent, te.mi))

        print('finished training', prefix)
        results = [test() for _ in range(32)]
        outputs[prefix] = {k: np.mean([r[k] for r in results])
                           for k in ['ent', 'mi']}

        tf.get_collection_ref('datasets').clear()
        print('finished benchmark for', prefix, outputs[prefix])
      count += 1

    with open('./output_%d.pkl' % rank, 'wb') as f:
      pickle.dump(outputs, f)
