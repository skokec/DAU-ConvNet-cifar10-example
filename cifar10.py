# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
from tensorflow.contrib import keras
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib import losses as losses_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope


from dau_conv import dau_conv2d
from dau_conv import DAUGridMean
from dau_conv import dau_conv2d_tf

import cifar10_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 256,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_string('backend', 'dau_conv',
                           """Version of the backend (cnn, dau_conv or dau_conv_tf.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 50.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9999999

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

def inference(images,in_training=True):
  if FLAGS.backend.lower() == 'cnn':
    return inference_vanilla_cnn(images,in_training)
  elif FLAGS.backend.lower() == 'dau_conv':
    return inference_dau_conv(images,in_training)
  elif FLAGS.backend.lower() == 'dau_conv_tf':
    return inference_dau_conv_tf(images,in_training)
  else:
    raise Exception('Invalid "backend" arg (%s)! Allowed only: "cnn", "dau_conv", and "dau_conv_tf"' % FLAGS.backend.lower())
  

def inference_dau_conv(images,in_training=True):
    inputs = images
    scope = ''
    with variable_scope.variable_scope(scope, 'cifar10', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, max_pool2d
    with arg_scope(
            [layers.conv2d, dau_conv2d, layers.fully_connected, layers_lib.max_pool2d, layers.batch_norm],
            outputs_collections=end_points_collection):

        # Apply specific parameters to all conv2d layers (to use batch norm and relu - relu is by default)
        with arg_scope([layers.conv2d, dau_conv2d, layers.fully_connected],
                       weights_regularizer=regularizers.l2_regularizer(0.1),
                       weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                       #weights_initializer= lambda shape,dtype=tf.float32, partition_info=None: tf.contrib.layers.xavier_initializer(uniform=False),
                       biases_initializer=None,
                       #normalizer_fn=layers.batch_norm,
                       #normalizer_params={'center': True,
                       #                   'scale': True,
                       #                   #'is_training': in_training,
                       #                   'decay': BATCHNORM_MOVING_AVERAGE_DECAY, # Decay for the moving averages.
                       #                   'epsilon': 0.001, # epsilon to prevent 0s in variance.
                       #                   'data_format':'NCHW',
                       #                },
                       normalizer_fn=tf.layers.batch_normalization,
                        normalizer_params=dict(center=True,
                                               scale=True,
                                               #momentum=BATCHNORM_MOVING_AVERAGE_DECAY, # Decay for the moving averages.
                                               epsilon=0.001, # epsilon to prevent 0s in variance.
                                               axis=1,
                                               training=in_training)):

            inputs = tf.transpose(inputs, [0,3,1,2])
            print("input: ",inputs.shape)
            net = layers_lib.repeat(inputs, 1, dau_conv2d, 96, dau_units=(2,2), max_kernel_size=17,
                                    mu2_initializer=DAUGridMean(dau_units=(2,2), max_value=4, dau_unit_axis=1),
                                    mu1_initializer=DAUGridMean(dau_units=(2,2), max_value=4, dau_unit_axis=2),
                                    #mu2_initializer=tf.constant_initializer(0),
                                    #mu1_initializer=tf.constant_initializer(0),
                                    dau_unit_border_bound=2.0,
                                    mu_learning_rate_factor=1, data_format='NCHW', scope='dau_conv1')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool1', data_format="NCHW")
            #'''
            net = layers_lib.repeat(net, 1, dau_conv2d, 96, dau_units=(2,2), max_kernel_size=17,
                                    mu2_initializer=DAUGridMean(dau_units=(2,2), max_value=4, dau_unit_axis=1),
                                    mu1_initializer=DAUGridMean(dau_units=(2,2), max_value=4, dau_unit_axis=2),
                                    #mu2_initializer=tf.constant_initializer(0),
                                    #mu1_initializer=tf.constant_initializer(0),
                                    dau_unit_border_bound=2.0,
                                    mu_learning_rate_factor=1, data_format='NCHW', scope='dau_conv2')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool2', data_format="NCHW")

            net = layers_lib.repeat(net, 1, dau_conv2d, 192, dau_units=(2,2), max_kernel_size=17,
                                    mu2_initializer=DAUGridMean(dau_units=(2,2), max_value=4, dau_unit_axis=1),
                                    mu1_initializer=DAUGridMean(dau_units=(2,2), max_value=4, dau_unit_axis=2),
                                    #mu2_initializer=tf.constant_initializer(0),
                                    #mu1_initializer=tf.constant_initializer(0),
                                    dau_unit_border_bound=2.0,
                                    mu_learning_rate_factor=1, data_format='NCHW', scope='dau_conv3')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool3', data_format="NCHW")
            #'''
            net = tf.reshape(net, [net.shape[0], -1])

            net = layers.fully_connected(net, NUM_CLASSES, scope='fc4',
                                         activation_fn=None,
                                         normalizer_fn=None,
                                         biases_initializer=tf.constant_initializer(0))


    return net

def inference_dau_conv_tf(images,in_training=True):
    inputs = images
    scope = ''
    with variable_scope.variable_scope(scope, 'cifar10', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, max_pool2d
    with arg_scope(
            [layers.conv2d, dau_conv2d_tf, layers.fully_connected, layers_lib.max_pool2d, layers.batch_norm],
            outputs_collections=end_points_collection):

        # Apply specific parameters to all conv2d layers (to use batch norm and relu - relu is by default)
        with arg_scope([layers.conv2d, dau_conv2d_tf, layers.fully_connected],
                       weights_regularizer=regularizers.l2_regularizer(0.1),
                       weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                       #weights_initializer= lambda shape,dtype=tf.float32, partition_info=None: tf.contrib.layers.xavier_initializer(uniform=False),
                       biases_initializer=None,
                       #normalizer_fn=layers.batch_norm,
                       #normalizer_params={'center': True,
                       #                   'scale': True,
                       #                   #'is_training': in_training,
                       #                   'decay': BATCHNORM_MOVING_AVERAGE_DECAY, # Decay for the moving averages.
                       #                   'epsilon': 0.001, # epsilon to prevent 0s in variance.
                       #                   'data_format':'NCHW',
                       #                },
                       normalizer_fn=tf.layers.batch_normalization,
                        normalizer_params=dict(center=True,
                                               scale=True,
                                               #momentum=BATCHNORM_MOVING_AVERAGE_DECAY, # Decay for the moving averages.
                                               epsilon=0.001, # epsilon to prevent 0s in variance.
                                               axis=-1,
                                               training=in_training)):

            
            print("input: ",inputs.shape)
            net = layers_lib.repeat(inputs, 1, dau_conv2d_tf, 96, dau_units=(2,2), max_kernel_size=17,
                                    mu2_initializer=DAUGridMean(dau_units=(2,2), max_value=4, dau_unit_axis=1),
                                    mu1_initializer=DAUGridMean(dau_units=(2,2), max_value=4, dau_unit_axis=2),
                                    #mu2_initializer=tf.constant_initializer(0),
                                    #mu1_initializer=tf.constant_initializer(0),
                                    dau_unit_border_bound=2.0,
                                    mu_learning_rate_factor=1, data_format='NHWC', scope='dau_conv1')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool1', data_format="NHWC")
            #'''
            net = layers_lib.repeat(net, 1, dau_conv2d_tf, 96, dau_units=(2,2), max_kernel_size=17,
                                    mu2_initializer=DAUGridMean(dau_units=(2,2), max_value=4, dau_unit_axis=1),
                                    mu1_initializer=DAUGridMean(dau_units=(2,2), max_value=4, dau_unit_axis=2),
                                    #mu2_initializer=tf.constant_initializer(0),
                                    #mu1_initializer=tf.constant_initializer(0),
                                    dau_unit_border_bound=2.0,
                                    mu_learning_rate_factor=1, data_format='NHWC', scope='dau_conv2')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool2', data_format="NHWC")

            net = layers_lib.repeat(net, 1, dau_conv2d_tf, 192, dau_units=(2,2), max_kernel_size=17,
                                    mu2_initializer=DAUGridMean(dau_units=(2,2), max_value=4, dau_unit_axis=1),
                                    mu1_initializer=DAUGridMean(dau_units=(2,2), max_value=4, dau_unit_axis=2),
                                    #mu2_initializer=tf.constant_initializer(0),
                                    #mu1_initializer=tf.constant_initializer(0),
                                    dau_unit_border_bound=2.0,
                                    mu_learning_rate_factor=1, data_format='NHWC', scope='dau_conv3')
            net = layers_lib.max_pool2d(net, [2, 2], scope='pool3', data_format="NHWC")
            #'''
            net = tf.reshape(net, [net.shape[0], -1])

            net = layers.fully_connected(net, NUM_CLASSES, scope='fc4',
                                         activation_fn=None,
                                         normalizer_fn=None,
                                         biases_initializer=tf.constant_initializer(0))


    return net


def inference_vanilla_cnn(images,in_training=True):
    inputs = images
    scope = ''
    with variable_scope.variable_scope(scope, 'cifar10', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, max_pool2d
    with arg_scope(
            [layers.conv2d, layers.fully_connected, layers_lib.max_pool2d, layers.batch_norm],
            outputs_collections=end_points_collection):

        # Apply specific parameters to all conv2d layers (to use batch norm and relu - relu is by default)
        with arg_scope([layers.conv2d, layers.fully_connected],
                       weights_regularizer=regularizers.l2_regularizer(0.0005),
                       weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                       biases_initializer=None,
                       normalizer_fn=tf.layers.batch_normalization,
                       normalizer_params=dict(center=True,
                                              scale=True,
                                              #momentum=BATCHNORM_MOVING_AVERAGE_DECAY, # Decay for the moving averages.
                                              epsilon=0.001, # epsilon to prevent 0s in variance.
                                              training=in_training)):

                net = layers_lib.repeat(inputs, 1, layers.conv2d, 32, [3, 3], scope='conv1')
                _activation_summary(net)
                net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')

                net = layers_lib.repeat(net, 1, layers.conv2d, 32, [3, 3], scope='conv2')
                _activation_summary(net)
                net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')

                net = layers_lib.repeat(net, 1, layers.conv2d, 32, [3, 3], scope='conv3')
                _activation_summary(net)
                net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
                net = tf.reshape(net, [net.shape[0], -1])

                net = layers.fully_connected(net, NUM_CLASSES, scope='fc4',
                                             activation_fn=None,
                                             normalizer_fn=None,
                                             biases_initializer=tf.constant_initializer(0))


    return net


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  print("labels",labels.shape)
  print("logits",logits.shape)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op] + update_ops):
    #opt = tf.train.GradientDescentOptimizer(lr)
    #opt = tf.train.MomentumOptimizer(lr, 0.9)
    opt = tf.train.AdamOptimizer(learning_rate=0.01)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  with tf.control_dependencies([apply_gradient_op]):
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

  return variables_averages_op


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
