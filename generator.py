import tensorflow as tf
import numpy as np
import sonnet as snt
import tf_utils as tfu
import argparse
import os
import os.path as osp
import pickle
import tcml_util as util

class Generator(snt.AbstractModule):
  def __init__(self):
    super(Generator, self).__init__()

    with self._enter_variable_scope():
      self._cores = tfu.Struct()
      layers_defs = [
          util.attention(keys_dim=128, vals_dim=32, query_dim=128),
          util.tc_block(32, [1,2,4,8,16]),
          util.attention(keys_dim=128, vals_dim=32, query_dim=128),
          util.tc_block(32, [1,2,4,8,16]),
          util.attention(keys_dim=128, vals_dim=32, query_dim=128),
          util.conv1x1(32),
      ]
      self.layers, self.layer_names = tfu.layer_factory(layer_defs)
      self._core = snt.Sequential(self.layers)


  def _build(self, obs):
    out = self._core(obs)
    return out


def train_generator(model, train_data, test_data):
  dataset = tf.data.Dataset(train_data).shuffle(512).batch(32)
  batch = dataset.make_one_shot_iterator().get_next()

  x = tf.placeholder(tf.float32, [32, None, 1])

  x_shifted = tf.concat([tf.zeros((32, 1)), x[:, :-1]], axis=1)

  x_hat = model(x_shifted)

  x_one_hot = tf.quantize_v2(x, -1, 1, 32)

  logits = tf.reduce_sum(x_hat * x_one_hot, axis=-1)

  




def main():
  data = dict()
  for itr in [0, 5, 10, 20, 1000, 10000]:
    file_loc = osp.join('data', '%d.pkl' % itr)
    with open(file_loc, 'rb') as f:
      data.update(pickle.load(f))


      
