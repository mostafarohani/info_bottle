import numpy as np
import sandbox.tcml.tf_utils as tfu


def attention(keys_dim, vals_dim, query_dim, max_path_length, dense=True, last_timestep_only=False):
  return dict(name='tfu.CausalAttention',
              kwargs=dict(keys_dim=keys_dim, vals_dim=vals_dim, query_dim=query_dim, num_heads=1,
                          buffer_size=max_path_length, dense=dense, last_timestep_only=last_timestep_only
                          )
              )


def tc(output_channels, dilation):
  return dict(name='tfu.CausalConv1D',
              kwargs=dict(train_initial_state=True, dense=True,
                          params=dict(kernel_shape=2,
                                      output_channels=output_channels,
                                      rate=rate
                                      )
                          )
              )


def tc_block(output_channels, dilations):
  return tfu._list_flatten([
      tc(output_channels, d) for d in dilations
  ])


def pool(mode, ksize=(2, 2), stride=(2, 2)):
  if isinstance(stride, int):
    stride = (stride, stride)
  if isinstance(ksize, int):
    ksize = (ksize, ksize)
  return dict(name='tfu.Pool2D',
              kwargs=dict(mode=mode, ksize=ksize, strides=stride)
              )


def dropout(rate):
  return dict(name='tf.nn.dropout',
              kwargs=dict(keep_prob=1 - rate)
              )


def conv1x1(output_dim):
  return dict(name='tfu.Conv1x1',
              kwargs=dict(output_size=output_dim)
              )


def batch_norm():
  return [
      dict(name='snt.BatchNorm',
           kwargs=dict(decay_rate=0.99,
                       scale=True,
                       regularizers=None,
                       fused=True)
           ),
      'tfu.leaky_relu',
  ]


def conv_2d(output_channels, kernel_shape=3, stride=1, rate=1,  padding='SAME'):
  return [
      dict(name='snt.Conv2D',
           kwargs=dict(output_channels=output_channels,
                       kernel_shape=kernel_shape,
                       stride=stride,
                       rate=rate,
                       padding=padding,
                       use_bias=False)  # pointless when followed by batchnorm
           ),
  ]


def separable_conv_2d(output_channels, channel_multiplier=1, kernel_shape=3, stride=1, padding='SAME'):
  return [
      dict(name='snt.SeparableConv2D',
           kwargs=dict(output_channels=output_channels,
                       channel_multiplier=channel_multiplier,
                       kernel_shape=kernel_shape,
                       stride=stride,
                       padding=padding,
                       use_bias=False)  # pointless when followed by batchnorm
           ),
      'tfu.leaky_relu',
      dict(name='snt.BatchNorm',
           kwargs=dict(decay_rate=0.99,
                       scale=True,
                       regularizers=None,
                       fused=True)
           ),
  ]


def conv_resblock(channel_dims, kernel_shape=3, separable=False):
  if separable:
    func = separable_conv_2d
  else:
    func = conv_2d
  layer_defs = tfu._list_flatten(
      [func(ch, kernel_shape=kernel_shape) for ch in channel_dims])
  return dict(name='tfu.ResBlock',
              kwargs=dict(params=layer_defs)
              )


def res_bottleneck(channel_dim, bottleneck_ratio=1, kernel_shape=3, stride=1):
  layer_defs = tfu._list_flatten([
      conv_2d(channel_dim, kernel_shape=1),
      batch_norm(),
      conv_2d(channel_dim, kernel_shape=kernel_shape),
      pool('max', ksize=[1, 1], stride=stride),
      batch_norm(),
      conv_2d(channel_dim * bottleneck_ratio, kernel_shape=1)
  ])
  return dict(name='tfu.ResBlock',
              kwargs=dict(params=layer_defs, shrink_factor=stride)
              )


def res_bottleneck_block(num_units, channel_dim, bottleneck_ratio=1, kernel_shape=3, stride=1):
  return [
      res_bottleneck(channel_dim, bottleneck_ratio, kernel_shape, stride=1)
  ] * (num_units - 1) + \
      [res_bottleneck(channel_dim, bottleneck_ratio, kernel_shape, stride)]


def basic_inception_module(channel_dims, stride=(2, 2)):
  path_one = tfu._list_flatten([
      conv1x1(channel_dims / 2),
      conv_2d(channel_dims / 2, stride=stride)
  ])
  path_two = tfu._list_flatten([
      conv1x1(channel_dims / 2),
      conv_2d(channel_dims / 2),
      conv_2d(channel_dims / 2, stride=stride)
  ])
  path_three = [pool(mode='max', ksize=stride, stride=stride)]

  return dict(name='tfu.InceptionModule',
              kwargs=dict(list_params=[path_one, path_two, path_three]))


def vectorize_conv_features(mode, output_size=None):
  """
  Has two modes,

  'flatten': takes a tensor of shape (B, W, L, C) and turns into (B, W*L*C)
  'global_mean': output_size must be given. takes a tensor of shape (B, W, L, C)
                 and uses 1x1 Conv to make it (B, W, L, output_size) and does a global
                 mean over the spatial dimensions, resulting in tensor of shape
                 (B, output_size)
  """
  if mode == 'flatten':
    # this behavior is hardcoded into the embedder so does not need to be specified here
    return []
  elif mode == 'global_mean':
    assert output_size != None, "output_size must be defined in mode 'global_mean'"
    return [
        conv1x1(output_size),
        dict(name='tf.reduce_mean',
             kwargs=dict(axis=[1, 2])
             )
    ]
  else:
    raise NotImplementedError(
        "mode must be either 'flatten' or 'global_mean'")


def construct_hp(
    feature_dim,
    embedding_dim,
    policy_layers,
    embedding_layers,
    max_n_class
):
  """
  Constructs the hyperparameters struct used for SNAIL models and their embedders

  Args:
    feature_dim: the dimension of the feature vectors produced by the embedding_layers
    embedding_dim: the dimension of the space to embed the label one_hot vector
    policy_layers: list of layers defining the SNAIL model
    embedding_layers: list of layers defining the feature extractor
    max_n_class: The max number of classes per episodes
  """
  return tfu.Struct.make({
      'embedding_dim': embedding_dim,
      'feature_dim': feature_dim,
      'opt_batch_size': 1,  # way input is structured, batch is in temporal dimension
      'output_type': dict(dim_y=max_n_class,
                          num_classes=max_n_class),
      'policy_params': dict(act=policy_layers),
      'embed_params': dict(layers=embedding_layers,
                           label_layers=[conv1x1(embedding_dim)]),
  })
