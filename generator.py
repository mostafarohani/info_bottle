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
          util.conv1x1(dim_out),
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
  #train_data = np.clip(train_data, lb, ub)
  def _build_model(data, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(512) \
                     .batch(batch_size) \
                     .prefetch(batch_size) \
                     .repeat()
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection('datasets', iterator.initializer)

    #iterator = dataset.make_one_shot_iterator()

    model = Generator(name, train_data.shape[-1], bins)

    x_raw = iterator.get_next()
   # print(np.linspace(lb, ub, bins))
    x_discr = tf.py_func(lambda x: np.digitize(x, np.linspace(lb, ub, bins)),
                         [x_raw], tf.int64)
    x_discr.set_shape(x_raw.shape)
    x = tf.one_hot(x_discr, bins)

    x_shifted = tf.concat([tf.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
    logits = model(x_shifted)
    logits = tf.clip_by_value(logits, -16, 16)

    ds = tf.contrib.distributions
    x_discr = tf.cast(x_discr, tf.float32)
   # logits = tf.Print(logits, [logits], "logits: ")

    log_prob = ds.Categorical(logits).log_prob(x_discr + 1e-3)
   # log_prob = tf.Print(log_prob, [log_prob], "log_prob: ")
    nll = tf.negative(tf.reduce_mean(log_prob))

    return nll

  train_nll = _build_model(train_data)
  train = tfu.make_train_op(train_nll, tfu.adam(lr=1e-4))
  train.stats = tfu.Struct(nll=train_nll)



  return train, _build_model(test_data, batch_size=128)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='data')
  parser.add_argument('--itr_count', type=int, default=10000)
  parser.add_argument('--print_window', type=int, default=100)
  parser.add_argument('--devices', type=str, default='0')


  FLAGS = parser.parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.devices
  all_iters = [0, 5, 10, 20, 1000, 9000]
  all_layers = [1, 3, 4]

  all_iters = [0]
  all_layers = [1]

  data = dict()
  for itr in all_iters:
    file_loc = osp.join(FLAGS.data_dir, '%d.pkl' % itr)
    with open(file_loc, 'rb') as f:
      data.update(pickle.load(f))

  with tf.Session().as_default() as sess:
    with tf.contrib.framework.arg_scope([snt.BatchNorm._build], is_training=True):
      train_ops = {}
      test_rollout = {}
      for itr in all_iters:
        for layer in all_layers:
          prefix = 'itr%d_layer%d' % (itr, layer)
          train_data = data['train_%s' % prefix]
          test_data = data['test_%s' % prefix]
          print('creating model for %s' % prefix)
          train_ops[prefix], test_rollout[prefix] = \
              train_generator(prefix, train_data, test_data)


      train = tfu.Function({}, train_ops)
      test = tfu.Function({}, test_rollout)

      for iterator_initializer in tf.get_collection('datasets'):
        sess.run(iterator_initializer)

      sess.run(tf.global_variables_initializer())
      tf.get_default_graph().finalize()

      for itr in range(FLAGS.itr_count):
        train()
        if itr % FLAGS.print_window == 0:
          train_output = train()
          test_output = test()
          print('======= itr %d =======' % itr)
          print('=== train ===')
          for k,v in train_output.items():
            print('%s nll %f' % (k, v.stats.nll))
          print('=== test ===')
          for k,v in test_output.items():
            print('%s nll %f' % (k, v))


  sess = tfu.Session(devices='0').__enter__()
  tfu.global_init(sess)
  sess.run(tf.get_collection('datasets'))
