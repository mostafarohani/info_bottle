import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import sonnet as snt
import tf_utils as tfu
import argparse
import os
import os.path as osp
import pickle

class MNISTClassifier(snt.AbstractModule):
  def __init__(self):
    super(MNISTClassifier, self).__init__()

    with self._enter_variable_scope():
      self._cores = tfu.Struct()
      self.layers = [
          snt.Linear(256),
          tf.tanh,
          snt.Linear(256),
          tf.tanh,
          snt.Linear(10)
      ]
      #self._cores.process = snt.Sequential(layers)


  def _build(self, obs):
    #vec_obs = snt.BatchFlatten()(obs)
    activations = []
    out = obs
    for layer in self.layers:
      out = layer(out)
      activations.append(out)
    return out, activations


def main():
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  with tf.Session().as_default() as sess:
    model = MNISTClassifier()
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    y_hat, activations = model(x)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

    train_op = tf.train.AdamOptimizer().minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    def save_activations(dump_dir, itr_num):
      highlights = [activations[1], activations[3], activations[4]]
      data = dict()
      for h, layer_num in zip(highlights, [1,3,4]):
        train_activations = sess.run(h, feed_dict={x : mnist.train.images, y : mnist.train.labels})
        test_activations = sess.run(h, feed_dict={x : mnist.test.images, y : mnist.test.labels})
        data['train_itr%d_layer%d' % (itr_num, layer_num)] = train_activations
        data['test_itr%d_layer%d' % (itr_num, layer_num)] = test_activations

      file_loc = osp.join(dump_dir, '%d.pkl' % itr_num)
      with open(file_loc, 'wb') as f:
        pickle.dump(data, f)

    sess.run(tf.global_variables_initializer())


    for itr in range(FLAGS.itr_count):
      batch = mnist.train.next_batch(50)
      if itr % 1000 == 0 or itr == 5 or itr == 10 or itr == 20:
        acc, l = sess.run([accuracy, loss], feed_dict={x : batch[0], y : batch[1]})
        print('itr % d, training accuracy %f, loss %f' % (itr, acc, l))
        save_activations(FLAGS.dump_dir, itr)
      else:
        sess.run([train_op], feed_dict={x : batch[0], y : batch[1]})

    test_acc = sess.run(accuracy, feed_dict={x : mnist.test.images, y : mnist.test.labels})

    print('test accuracy %f' % test_acc)





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data')
  parser.add_argument('--dump_dir', type=str,
                      default='data')
  parser.add_argument('--itr_count', type=int, default=10000)
  parser.add_argument('--devices', type=str, default='0')


  FLAGS = parser.parse_args()
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.devices
  main()
