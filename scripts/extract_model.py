# vim: expandtab:ts=4:sw=4
import pickle
import os
import sys
import errno
import argparse
import numpy as np
import cv2

import tensorflow as tf
import tensorflow.contrib.slim as slim

from easydict import EasyDict as edict
sys.path.append(os.path.join(os.path.dirname(__file__), 'proto'))
from proto.layer_pb2 import Layer  # NOQA
from proto.float_array_pb2 import FloatArray  # NOQA

#            input = session.graph.get_tensor_by_name('map/TensorArrayStack/TensorArrayGatherV3:0')
#
#            # conv1_1
#            conv1_1_weights = session.graph.get_tensor_by_name('conv1_1/weights/read:0')
#            # save_layer_output('conv1_1_weights', session.run(conv1_1_weights, feed_dict=x))
#            conv1_1_conv2d = session.graph.get_tensor_by_name('conv1_1/Conv2D:0')
#            # save_layer_output('conv1_1_conv2d', session.run(conv1_1_conv2d, feed_dict=x))
#
#            conv1_1_const = session.graph.get_tensor_by_name('conv1_1/conv1_1/bn/Const:0')
#            conv1_1_beta  = session.graph.get_tensor_by_name('conv1_1/conv1_1/bn/beta/read:0')
#            conv1_1_mean  = session.graph.get_tensor_by_name('conv1_1/conv1_1/bn/moving_mean/read:0')
#            conv1_1_var   = session.graph.get_tensor_by_name('conv1_1/conv1_1/bn/moving_variance/read:0')
#            conv1_1_fbn = session.graph.get_tensor_by_name('conv1_1/conv1_1/bn/FusedBatchNorm:0')
#
#            conv1_1_output = session.graph.get_tensor_by_name('conv1_1/Elu:0')
#
#            # conv1_2
#            conv1_2_weights = session.graph.get_tensor_by_name('conv1_2/weights/read:0')
#            conv1_2_conv2d = session.graph.get_tensor_by_name('conv1_2/Conv2D:0')
#            
#            conv1_2_const = session.graph.get_tensor_by_name('conv1_2/conv1_2/bn/Const:0')
#            conv1_2_beta  = session.graph.get_tensor_by_name('conv1_2/conv1_2/bn/beta/read:0')
#            conv1_2_mean  = session.graph.get_tensor_by_name('conv1_2/conv1_2/bn/moving_mean/read:0')
#            conv1_2_var   = session.graph.get_tensor_by_name('conv1_2/conv1_2/bn/moving_variance/read:0')
#            conv1_2_fbn = session.graph.get_tensor_by_name('conv1_2/conv1_2/bn/FusedBatchNorm:0')
#
#            conv1_2_output = session.graph.get_tensor_by_name('conv1_2/Elu:0')
#
#            # Pool1
#            pool1_max_pool = session.graph.get_tensor_by_name('pool1/MaxPool:0')
#
#            # conv2_1
#            conv2_1_1_weights = session.graph.get_tensor_by_name('conv2_1/1/weights/read:0')
#            conv2_1_1_conv2d = session.graph.get_tensor_by_name('conv2_1/1/Conv2D:0')
#
#            conv2_1_1_const = session.graph.get_tensor_by_name('conv2_1/1/conv2_1/1/bn/Const:0')
#            conv2_1_1_beta  = session.graph.get_tensor_by_name('conv2_1/1/conv2_1/1/bn/beta/read:0')
#            conv2_1_1_mean  = session.graph.get_tensor_by_name('conv2_1/1/conv2_1/1/bn/moving_mean/read:0')
#            conv2_1_1_var   = session.graph.get_tensor_by_name('conv2_1/1/conv2_1/1/bn/moving_variance/read:0')
#            conv2_1_1_fbn = session.graph.get_tensor_by_name('conv2_1/1/conv2_1/1/bn/FusedBatchNorm:0')
#           
#            conv2_1_1_output = session.graph.get_tensor_by_name('conv2_1/1/Elu:0')
#
#            conv2_1_2_weights = session.graph.get_tensor_by_name('conv2_1/2/weights/read:0')
#            conv2_1_2_conv2d = session.graph.get_tensor_by_name('conv2_1/2/Conv2D:0')
#            conv2_1_2_biases = session.graph.get_tensor_by_name('conv2_1/2/biases/read:0')
#            conv2_1_2_biasadd = session.graph.get_tensor_by_name('conv2_1/2/BiasAdd:0')
#
#            # pool1/MaxPool + conv2_1/2/BiasAdd
#            add = session.graph.get_tensor_by_name('add:0')
#
#            # conv2_3
#            conv2_3_bn_const = session.graph.get_tensor_by_name('conv2_3/bn/Const:0')
#            conv2_3_bn_beta  = session.graph.get_tensor_by_name('conv2_3/bn/beta/read:0')
#            conv2_3_bn_mean  = session.graph.get_tensor_by_name('conv2_3/bn/moving_mean/read:0')
#            conv2_3_bn_var   = session.graph.get_tensor_by_name('conv2_3/bn/moving_variance/read:0')
#            conv2_3_bn_fbn = session.graph.get_tensor_by_name('conv2_3/bn/FusedBatchNorm:0')           
#            elu = session.graph.get_tensor_by_name('Elu:0')
#
#            conv2_3_1_weights = session.graph.get_tensor_by_name('conv2_3/1/weights/read:0')
#            conv2_3_1_conv2d = session.graph.get_tensor_by_name('conv2_3/1/Conv2D:0')
#            conv2_3_1_const = session.graph.get_tensor_by_name('conv2_3/1/conv2_3/1/bn/Const:0')
#            conv2_3_1_beta  = session.graph.get_tensor_by_name('conv2_3/1/conv2_3/1/bn/beta/read:0')
#            conv2_3_1_mean  = session.graph.get_tensor_by_name('conv2_3/1/conv2_3/1/bn/moving_mean/read:0')
#            conv2_3_1_var   = session.graph.get_tensor_by_name('conv2_3/1/conv2_3/1/bn/moving_variance/read:0')
#            conv2_3_1_fbn = session.graph.get_tensor_by_name('conv2_3/1/conv2_3/1/bn/FusedBatchNorm:0')
#
#            conv2_3_1_elu = session.graph.get_tensor_by_name('conv2_3/1/Elu:0')
#
#            conv2_3_2_weights = session.graph.get_tensor_by_name('conv2_3/2/weights/read:0')
#            conv2_3_2_conv2d = session.graph.get_tensor_by_name('conv2_3/2/Conv2D:0')
#
#            conv2_3_2_biases = session.graph.get_tensor_by_name('conv2_3/2/biases/read:0')
#            conv2_3_2_biasadd = session.graph.get_tensor_by_name('conv2_3/2/BiasAdd:0')
#
#            # add + conv2_3/2/BiasAdd
#            add_1 = session.graph.get_tensor_by_name('add_1:0')
#
#            # conv3_1
#            conv3_1_bn_const = session.graph.get_tensor_by_name('conv3_1/bn/Const:0')
#            conv3_1_bn_beta  = session.graph.get_tensor_by_name('conv3_1/bn/beta/read:0')
#            conv3_1_bn_mean  = session.graph.get_tensor_by_name('conv3_1/bn/moving_mean/read:0')
#            conv3_1_bn_var   = session.graph.get_tensor_by_name('conv3_1/bn/moving_variance/read:0')
#            conv3_1_bn_fbn = session.graph.get_tensor_by_name('conv3_1/bn/FusedBatchNorm:0')
#
#            elu_1 = session.graph.get_tensor_by_name('Elu_1:0')
#
#            conv3_1_1_weights = session.graph.get_tensor_by_name('conv3_1/1/weights/read:0')
#            conv3_1_1_conv2d = session.graph.get_tensor_by_name('conv3_1/1/Conv2D:0')
#            conv3_1_1_const = session.graph.get_tensor_by_name('conv3_1/1/conv3_1/1/bn/Const:0')
#            conv3_1_1_beta  = session.graph.get_tensor_by_name('conv3_1/1/conv3_1/1/bn/beta/read:0')
#            conv3_1_1_mean  = session.graph.get_tensor_by_name('conv3_1/1/conv3_1/1/bn/moving_mean/read:0')
#            conv3_1_1_var   = session.graph.get_tensor_by_name('conv3_1/1/conv3_1/1/bn/moving_variance/read:0')
#            conv3_1_1_fbn = session.graph.get_tensor_by_name('conv3_1/1/conv3_1/1/bn/FusedBatchNorm:0')
#
#            conv3_1_1_elu = session.graph.get_tensor_by_name('conv3_1/1/Elu:0')
#
#            conv3_1_2_weights = session.graph.get_tensor_by_name('conv3_1/2/weights/read:0')
#            conv3_1_2_conv2d = session.graph.get_tensor_by_name('conv3_1/2/Conv2D:0')
#
#            conv3_1_2_biases = session.graph.get_tensor_by_name('conv3_1/2/biases/read:0')
#            conv3_1_2_biasadd = session.graph.get_tensor_by_name('conv3_1/2/BiasAdd:0')
#
#            conv3_1_projection_weights = session.graph.get_tensor_by_name('conv3_1/projection/weights/read:0')
#            # conv_3_1_projection_conv2d = add_1 * conv3_1_projection_weights
#            conv3_1_projection_conv2d = session.graph.get_tensor_by_name('conv3_1/projection/Conv2D:0')
#            # conv3_1/projection/Conv2D + conv3_1/2/BiasAdd
#            add_2 = session.graph.get_tensor_by_name('add_2:0')
#
#            # conv3_3
#            conv3_3_bn_const = session.graph.get_tensor_by_name('conv3_3/bn/Const:0')
#            conv3_3_bn_beta  = session.graph.get_tensor_by_name('conv3_3/bn/beta/read:0')
#            conv3_3_bn_mean  = session.graph.get_tensor_by_name('conv3_3/bn/moving_mean/read:0')
#            conv3_3_bn_var   = session.graph.get_tensor_by_name('conv3_3/bn/moving_variance/read:0')
#            conv3_3_bn_fbn = session.graph.get_tensor_by_name('conv3_3/bn/FusedBatchNorm:0')
#
#            elu_2 = session.graph.get_tensor_by_name('Elu_2:0')
#
#            conv3_3_1_weights = session.graph.get_tensor_by_name('conv3_3/1/weights/read:0')
#            conv3_3_1_conv2d = session.graph.get_tensor_by_name('conv3_3/1/Conv2D:0')
#
#            conv3_3_1_const = session.graph.get_tensor_by_name('conv3_3/1/conv3_3/1/bn/Const:0')
#            conv3_3_1_beta  = session.graph.get_tensor_by_name('conv3_3/1/conv3_3/1/bn/beta/read:0')
#            conv3_3_1_mean  = session.graph.get_tensor_by_name('conv3_3/1/conv3_3/1/bn/moving_mean/read:0')
#            conv3_3_1_var   = session.graph.get_tensor_by_name('conv3_3/1/conv3_3/1/bn/moving_variance/read:0')
#            conv3_3_1_fbn = session.graph.get_tensor_by_name('conv3_3/1/conv3_3/1/bn/FusedBatchNorm:0')
#
#            conv3_3_1_elu = session.graph.get_tensor_by_name('conv3_3/1/Elu:0')
#
#            conv3_3_2_weights = session.graph.get_tensor_by_name('conv3_3/2/weights/read:0')
#            conv3_3_2_conv2d = session.graph.get_tensor_by_name('conv3_3/2/Conv2D:0')
#
#            conv3_3_2_biases = session.graph.get_tensor_by_name('conv3_3/2/biases/read:0')
#            conv3_3_2_biasadd = session.graph.get_tensor_by_name('conv3_3/2/BiasAdd:0')
#
#            # add_2 + conv3_3/2/BiasAdd
#            add_3 = session.graph.get_tensor_by_name('add_3:0')
#
#             # conv4_1
#            conv4_1_bn_const = session.graph.get_tensor_by_name('conv4_1/bn/Const:0')
#            conv4_1_bn_beta  = session.graph.get_tensor_by_name('conv4_1/bn/beta/read:0')
#            conv4_1_bn_mean  = session.graph.get_tensor_by_name('conv4_1/bn/moving_mean/read:0')
#            conv4_1_bn_var   = session.graph.get_tensor_by_name('conv4_1/bn/moving_variance/read:0')
#            conv4_1_bn_fbn = session.graph.get_tensor_by_name('conv4_1/bn/FusedBatchNorm:0')
#
#            elu_3 = session.graph.get_tensor_by_name('Elu_3:0')
#
#            conv4_1_1_weights = session.graph.get_tensor_by_name('conv4_1/1/weights/read:0')
#            conv4_1_1_conv2d = session.graph.get_tensor_by_name('conv4_1/1/Conv2D:0')
#            conv4_1_1_const = session.graph.get_tensor_by_name('conv4_1/1/conv4_1/1/bn/Const:0')
#            conv4_1_1_beta  = session.graph.get_tensor_by_name('conv4_1/1/conv4_1/1/bn/beta/read:0')
#            conv4_1_1_mean  = session.graph.get_tensor_by_name('conv4_1/1/conv4_1/1/bn/moving_mean/read:0')
#            conv4_1_1_var   = session.graph.get_tensor_by_name('conv4_1/1/conv4_1/1/bn/moving_variance/read:0')
#            conv4_1_1_fbn = session.graph.get_tensor_by_name('conv4_1/1/conv4_1/1/bn/FusedBatchNorm:0')
#
#            conv4_1_1_elu = session.graph.get_tensor_by_name('conv4_1/1/Elu:0')
#
#            conv4_1_2_weights = session.graph.get_tensor_by_name('conv4_1/2/weights/read:0')
#            conv4_1_2_conv2d = session.graph.get_tensor_by_name('conv4_1/2/Conv2D:0')
#
#            conv4_1_2_biases = session.graph.get_tensor_by_name('conv4_1/2/biases/read:0')
#            conv4_1_2_biasadd = session.graph.get_tensor_by_name('conv4_1/2/BiasAdd:0')
#
#            conv4_1_projection_weights = session.graph.get_tensor_by_name('conv4_1/projection/weights/read:0')
#            # conv_4_1_projection_conv2d = add_3 * conv4_1_projection_weights
#            conv4_1_projection_conv2d = session.graph.get_tensor_by_name('conv4_1/projection/Conv2D:0')
#            # conv4_1/projection/Conv2D + conv4_1/2/BiasAdd
#            add_4 = session.graph.get_tensor_by_name('add_4:0')
#
#            # conv4_3
#            conv4_3_bn_const = session.graph.get_tensor_by_name('conv4_3/bn/Const:0')
#            conv4_3_bn_beta  = session.graph.get_tensor_by_name('conv4_3/bn/beta/read:0')
#            conv4_3_bn_mean  = session.graph.get_tensor_by_name('conv4_3/bn/moving_mean/read:0')
#            conv4_3_bn_var   = session.graph.get_tensor_by_name('conv4_3/bn/moving_variance/read:0')
#            conv4_3_bn_fbn = session.graph.get_tensor_by_name('conv4_3/bn/FusedBatchNorm:0')
#
#            elu_4 = session.graph.get_tensor_by_name('Elu_4:0')
#
#            conv4_3_1_weights = session.graph.get_tensor_by_name('conv4_3/1/weights/read:0')
#            conv4_3_1_conv2d = session.graph.get_tensor_by_name('conv4_3/1/Conv2D:0')
#
#            conv4_3_1_const = session.graph.get_tensor_by_name('conv4_3/1/conv4_3/1/bn/Const:0')
#            conv4_3_1_beta  = session.graph.get_tensor_by_name('conv4_3/1/conv4_3/1/bn/beta/read:0')
#            conv4_3_1_mean  = session.graph.get_tensor_by_name('conv4_3/1/conv4_3/1/bn/moving_mean/read:0')
#            conv4_3_1_var   = session.graph.get_tensor_by_name('conv4_3/1/conv4_3/1/bn/moving_variance/read:0')
#            conv4_3_1_fbn = session.graph.get_tensor_by_name('conv4_3/1/conv4_3/1/bn/FusedBatchNorm:0')
#
#            conv4_3_1_elu = session.graph.get_tensor_by_name('conv4_3/1/Elu:0')
#
#            conv4_3_2_weights = session.graph.get_tensor_by_name('conv4_3/2/weights/read:0')
#            conv4_3_2_conv2d = session.graph.get_tensor_by_name('conv4_3/2/Conv2D:0')
#
#            conv4_3_2_biases = session.graph.get_tensor_by_name('conv4_3/2/biases/read:0')
#            conv4_3_2_biasadd = session.graph.get_tensor_by_name('conv4_3/2/BiasAdd:0')
#
#            # add_4 + conv4_3/2/BiasAdd
#            add_5 = session.graph.get_tensor_by_name('add_5:0')
#
#            # flatten shape = shape(add_5)
#            flatten_shape = session.graph.get_tensor_by_namen('Flatten/flatten/Shape:0')
#            flatten_strided_slice_stack = session.graph.get_tensor_by_name('Flatten/flatten/strided_slice/stack:0')
#            flatten_strided_slice_stack_1 = session.graph.get_tensor_by_name('Flatten/flatten/strided_slice/stack_1:0')
#            flatten_strided_slice_stack_2 = session.graph.get_tensor_by_name('Flatten/flatten/strided_slice/stack_2:0')
#            # StridedSlice(flatten_shape, flatten_strided_slice_stack, flatten_strided_slice_stack_1, flatten_strided_slice_stack_2)
#            flatten_strided_slice = session.graph.get_tensor_by_name('Flatten/flatten/strided_slice:0')
#            flatten_reshape_shape_1 = session.graph.get_tensor_by_name('Flatten/flatten/Reshape/shape/1:0')
#            # Pack (flatten_strided_slice, flatten_reshape_shape_1)
#            flatten_reshape_shape = session.graph.get_tensor_by_name('Flatten/flatten/Reshape/shape:0')
#            # Reshape = reshape(add_5, flatten_reshape_shape)
#            flatten_reshape = session.graph.get_tensor_by_name('Flatten/flatten/Reshape:0')
#
#            # MatMul(flatten_reshape, fc1_weights)
#            fc1_weights = session.graph.get_tensor_by_name('fc1/weights/read:0')
#            fc1_matmul = session.graph.get_tensor_by_name('fc1/MatMul:0')
#
#            fc1_fc1_bn_reshape_shape = session.graph.get_tensor_by_name('fc1/fc1/bn/Reshape/shape:0')
#            # Reshape(fc1_matmul, fc1_fc1_bn_reshape_shape)
#            fc1_fc1_bn_reshape = session.graph.get_tensor_by_name('fc1/fc1/bn/Reshape:0')
#            fc1_fc1_bn_const = session.graph.get_tensor_by_name('fc1/fc1/bn/Const:0')
#            fc1_fc1_bn_beta = session.graph.get_tensor_by_name('fc1/fc1/bn/beta/read:0')
#            fc1_fc1_bn_mean = session.graph.get_tensor_by_name('fc1/fc1/bn/moving_mean/read:0')
#            fc1_fc1_bn_var = session.graph.get_tensor_by_name('fc1/fc1/bn/moving_variance/read:0')
#
#            # FusedBatchNorm(fc1_fc1_bn_reshape, fc1_fc1_bn_const, fc1_fc1_bn_beta, fc1_fc1_bn_mean, fc1_fc1_bn_var)
#            fc1_fc1_bn_fbn = session.graph.get_tensor_by_name('fc1/fc1/bn/FusedBatchNorm:0')
#            # shape(fc1_matmul)
#            fc1_fc1_bn_shape = session.graph.get_tensor_by_name('fc1/fc1/bn/Shape:0')
#            # reshape(fc1_fc1_bn_shape, fc1_fc1_bn_fbn)
#            fc1_fc1_bn_reshape_1 = session.graph.get_tensor_by_name('fc1/fc1/bn/Reshape_1:0')
#            # elu(fc1_fc1_bn_reshape_1)
#            fc1_elu = session.graph.get_tensor_by_name('fc1/Elu:0')
#
#            # ball
#            ball_reshape_shape = session.graph.get_tensor_by_name('ball/Reshape/shape:0')
#            # ball_rashape = Reshape(fc1_elu,ball/Reshape/shape)
#            ball_reshape = session.graph.get_tensor_by_name('ball/Reshape:0')
#            ball_const = session.graph.get_tensor_by_name('ball/Const:0')
#            ball_beta = session.graph.get_tensor_by_name('ball/beta/read:0')
#            ball_mean = session.graph.get_tensor_by_name('ball/moving_mean/read:0')
#            ball_var = session.graph.get_tensor_by_name('ball/moving_variance/read:0')
#            # fused_batch_norm(ball_reshape, ball_const, ball_beta, ball_mean, ball_var)
#            ball_fbn = session.graph.get_tensor_by_name('ball/FusedBatchNorm:0')
#
#            # shape(fc1_elu)
#            ball_shape = session.graph.get_tensor_by_name('ball/Shape:0')
#            # reshape(ball_fbn, ball_shape)
#            ball_reshape_1 = session.graph.get_tensor_by_name('ball/Reshape_1:0')
#            # square(ball_reshape_1)
#            square = session.graph.get_tensor_by_name('Square:0')
#            sum_reduction_indices = session.graph.get_tensor_by_name('Sum/reduction_indices:0')
#            # sum = sum(square, sum_reduction_indices)
#            sum = session.graph.get_tensor_by_name('Sum:0')
#            const = session.graph.get_tensor_by_name('Const:0')
#            add_6 = session.graph.get_tensor_by_name('add_6:0')
#            sqrt = session.graph.get_tensor_by_name('Sqrt:0')
#            # truediv = ball_reshape_1/sqrt
#            out = session.graph.get_tensor_by_name('truediv:0')
#
#            conv2_1_nodes = {
#                'input': input,
#                'conv2_1_1_weights': conv2_1_1_weights,
#                'conv2_1_1_conv2d': conv2_1_1_conv2d,
#                'conv2_1_1_const': conv2_1_1_const,
#                'conv2_1_1_beta': conv2_1_1_beta,
#                'conv2_1_1_mean': conv2_1_1_mean,
#                'conv2_1_1_var': conv2_1_1_var,
#                'conv2_1_1_fbn': conv2_1_1_fbn,
#                'conv2_1_1_output': conv2_1_1_output,
#                'conv2_1_2_weights': conv2_1_2_weights, 
#                'conv2_1_2_conv2d': conv2_1_2_conv2d,
#                'conv2_1_2_biases': conv2_1_2_biases, 
#                'conv2_1_2_biasadd': conv2_1_2_biasadd,
#                'fc1_elu': fc1_elu,
#                'out': out,
#                'result': feature_var
#            }
#
#            import pdb; pdb.set_trace()
#            conv2_1 = session.run(conv2_1_nodes, feed_dict=x)
#            pickle.dump(conv2_1, open('pkl/conv2_1.pkl', 'wb'))
#            return conv2_1['result']            

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def _batch_norm_fn(x, scope=None):
    if scope is None:
        scope = tf.get_variable_scope().name + "/bn"
    return slim.batch_norm(x, scope=scope)


def create_link(
        incoming, network_builder, scope, nonlinearity=tf.nn.elu,
        weights_initializer=tf.truncated_normal_initializer(stddev=1e-3),
        regularizer=None, is_first=False, summarize_activations=True):
    if is_first:
        network = incoming
    else:
        network = _batch_norm_fn(incoming, scope=scope + "/bn")
        network = nonlinearity(network)
        if summarize_activations:
            tf.summary.histogram(scope+"/activations", network)

    pre_block_network = network
    post_block_network = network_builder(pre_block_network, scope)

    incoming_dim = pre_block_network.get_shape().as_list()[-1]
    outgoing_dim = post_block_network.get_shape().as_list()[-1]
    if incoming_dim != outgoing_dim:
        assert outgoing_dim == 2 * incoming_dim, \
            "%d != %d" % (outgoing_dim, 2 * incoming)
        projection = slim.conv2d(
            incoming, outgoing_dim, 1, 2, padding="SAME", activation_fn=None,
            scope=scope+"/projection", weights_initializer=weights_initializer,
            biases_initializer=None, weights_regularizer=regularizer)
        network = projection + post_block_network
    else:
        network = incoming + post_block_network
    return network


def create_inner_block(
        incoming, scope, nonlinearity=tf.nn.elu,
        weights_initializer=tf.truncated_normal_initializer(1e-3),
        bias_initializer=tf.zeros_initializer(), regularizer=None,
        increase_dim=False, summarize_activations=True):
    n = incoming.get_shape().as_list()[-1]
    stride = 1
    if increase_dim:
        n *= 2
        stride = 2

    incoming = slim.conv2d(
        incoming, n, [3, 3], stride, activation_fn=nonlinearity, padding="SAME",
        normalizer_fn=_batch_norm_fn, weights_initializer=weights_initializer,
        biases_initializer=bias_initializer, weights_regularizer=regularizer,
        scope=scope + "/1")
    if summarize_activations:
        tf.summary.histogram(incoming.name + "/activations", incoming)

    incoming = slim.dropout(incoming, keep_prob=0.6)

    incoming = slim.conv2d(
        incoming, n, [3, 3], 1, activation_fn=None, padding="SAME",
        normalizer_fn=None, weights_initializer=weights_initializer,
        biases_initializer=bias_initializer, weights_regularizer=regularizer,
        scope=scope + "/2")
    return incoming


def residual_block(incoming, scope, nonlinearity=tf.nn.elu,
                   weights_initializer=tf.truncated_normal_initializer(1e3),
                   bias_initializer=tf.zeros_initializer(), regularizer=None,
                   increase_dim=False, is_first=False,
                   summarize_activations=True):

    def network_builder(x, s):
        return create_inner_block(
            x, s, nonlinearity, weights_initializer, bias_initializer,
            regularizer, increase_dim, summarize_activations)

    return create_link(
        incoming, network_builder, scope, nonlinearity, weights_initializer,
        regularizer, is_first, summarize_activations)


def _create_network(incoming, num_classes, reuse=None, l2_normalize=True,
                   create_summaries=True, weight_decay=1e-8):
    nonlinearity = tf.nn.elu
    conv_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
    conv_bias_init = tf.zeros_initializer()
    conv_regularizer = slim.l2_regularizer(weight_decay)
    fc_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
    fc_bias_init = tf.zeros_initializer()
    fc_regularizer = slim.l2_regularizer(weight_decay)

    def batch_norm_fn(x):
        return slim.batch_norm(x, scope=tf.get_variable_scope().name + "/bn")

    network = incoming
    network = slim.conv2d(
        network, 32, [3, 3], stride=1, activation_fn=nonlinearity,
        padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_1",
        weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
        weights_regularizer=conv_regularizer)
    if create_summaries:
        tf.summary.histogram(network.name + "/activations", network)
        tf.summary.image("conv1_1/weights", tf.transpose(
            slim.get_variables("conv1_1/weights:0")[0], [3, 0, 1, 2]),
                         max_images=128)
    network = slim.conv2d(
        network, 32, [3, 3], stride=1, activation_fn=nonlinearity,
        padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_2",
        weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
        weights_regularizer=conv_regularizer)
    if create_summaries:
        tf.summary.histogram(network.name + "/activations", network)

    # NOTE(nwojke): This is missing a padding="SAME" to match the CNN
    # architecture in Table 1 of the paper. Information on how this affects
    # performance on MOT 16 training sequences can be found in
    # issue 10 https://github.com/nwojke/deep_sort/issues/10
    network = slim.max_pool2d(network, [3, 3], [2, 2], scope="pool1")

    network = residual_block(
        network, "conv2_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False, is_first=True,
        summarize_activations=create_summaries)
    network = residual_block(
        network, "conv2_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    network = residual_block(
        network, "conv3_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=True,
        summarize_activations=create_summaries)
    network = residual_block(
        network, "conv3_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    network = residual_block(
        network, "conv4_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=True,
        summarize_activations=create_summaries)
    network = residual_block(
        network, "conv4_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    feature_dim = network.get_shape().as_list()[-1]
    print("feature dimensionality: ", feature_dim)
    network = slim.flatten(network)

    network = slim.dropout(network, keep_prob=0.6)
    network = slim.fully_connected(
        network, feature_dim, activation_fn=nonlinearity,
        normalizer_fn=batch_norm_fn, weights_regularizer=fc_regularizer,
        scope="fc1", weights_initializer=fc_weight_init,
        biases_initializer=fc_bias_init)

    features = network

    if l2_normalize:
        # Features in rows, normalize axis 1.
        features = slim.batch_norm(features, scope="ball", reuse=reuse)
        feature_norm = tf.sqrt(
            tf.constant(1e-8, tf.float32) +
            tf.reduce_sum(tf.square(features), [1], keep_dims=True))
        features = features / feature_norm

        with slim.variable_scope.variable_scope("ball", reuse=reuse):
            weights = slim.model_variable(
                "mean_vectors", (feature_dim, num_classes),
                initializer=tf.truncated_normal_initializer(stddev=1e-3),
                regularizer=None)
            scale = slim.model_variable(
                "scale", (num_classes, ), tf.float32,
                tf.constant_initializer(0., tf.float32), regularizer=None)
            if create_summaries:
                tf.summary.histogram("scale", scale)
            # scale = slim.model_variable(
            #     "scale", (), tf.float32,
            #     initializer=tf.constant_initializer(0., tf.float32),
            #     regularizer=slim.l2_regularizer(1e-2))
            # if create_summaries:
            #     tf.scalar_summary("scale", scale)
            scale = tf.nn.softplus(scale)

        # Each mean vector in columns, normalize axis 0.
        weight_norm = tf.sqrt(
            tf.constant(1e-8, tf.float32) +
            tf.reduce_sum(tf.square(weights), [0], keep_dims=True))
        logits = scale * tf.matmul(features, weights / weight_norm)

    else:
        logits = slim.fully_connected(
            features, num_classes, activation_fn=None,
            normalizer_fn=None, weights_regularizer=fc_regularizer,
            scope="softmax", weights_initializer=fc_weight_init,
            biases_initializer=fc_bias_init)

    return features, logits


def _network_factory(num_classes, is_training, weight_decay=1e-8):

    def factory_fn(image, reuse, l2_normalize):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=is_training):
                with slim.arg_scope([slim.conv2d, slim.fully_connected,
                                     slim.batch_norm, slim.layer_norm],
                                    reuse=reuse):
                    features, logits = _create_network(
                        image, num_classes, l2_normalize=l2_normalize,
                        reuse=reuse, create_summaries=is_training,
                        weight_decay=weight_decay)
                    return features, logits

    return factory_fn


def _preprocess(image, is_training=False, enable_more_augmentation=True):
    image = image[:, :, ::-1]  # BGR to RGB
    if is_training:
        image = tf.image.random_flip_left_right(image)
        if enable_more_augmentation:
            image = tf.image.random_brightness(image, max_delta=50)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    return image


def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, patch_shape[::-1])

    return image

def shape(arr):
    shape = [1, 1, 1, 1]
    shape[-len(arr.shape):] = arr.shape
    return shape

def save_layer(node, f, layer_name, weights=None, bias=None):
    layer = Layer()
    layer.name = layer_name

    if weights is not None:
        print('Saving weights %s\t%s' % (layer_name, weights.shape))
        layer.shape.d1, layer.shape.d2, layer.shape.d3, layer.shape.d4 = shape(weights)        
    elif bias is not None:
        layer.shape.d1, layer.shape.d2, layer.shape.d3, layer.shape.d4 = shape(bias)

    if weights is not None:
        for value in weights.flatten():
            layer.weights.append(float(value))

    if bias is not None:
        for value in bias.flatten():
            layer.bias.append(float(value))

    f.write(layer.SerializeToString())


def save_layer_conv(node, tf_nodes):
    # save weights
    basedir = os.path.dirname(os.path.realpath(__file__))
    filedir = os.path.join(basedir, 'resources/networks/layers')
    filepath = os.path.join(filedir, node.name)
    mkdir_p(filedir)

    weights = tf_nodes.weights.transpose((3, 2, 0, 1))
    with open(filepath, 'wb') as f:
        save_layer(node, f, node.name, weights, None)

def save_layer_convbias(node, tf_nodes):
    # save weights
    basedir = os.path.dirname(os.path.realpath(__file__))
    filedir = os.path.join(basedir, 'resources/networks/layers')
    filepath = os.path.join(filedir, node.name)
    mkdir_p(filedir)

    weights = tf_nodes.weights.transpose((3, 2, 0, 1))
    bias = tf_nodes.bias
    with open(filepath, 'wb') as f:
        save_layer(node, f, node.name, weights, bias)

        
def save_layer_conv_fbn(node, tf_nodes):
    basedir = os.path.dirname(os.path.realpath(__file__))
    filedir = os.path.join(basedir, 'resources/networks/layers')
    filepath = os.path.join(filedir, node.name)
    mkdir_p(filedir)

    weights = tf_nodes.weights.transpose(3, 2, 0, 1)
    bias = np.hstack((tf_nodes.beta, tf_nodes.mean, tf_nodes.variance))
    with open(filepath, 'wb') as f:
        save_layer(node, f, node.name, weights, bias)

def save_layer_fbn(node, tf_nodes):
    basedir = os.path.dirname(os.path.realpath(__file__))
    filedir = os.path.join(basedir, 'resources/networks/layers')
    filepath = os.path.join(filedir, node.name)
    mkdir_p(filedir)

    bias = np.hstack((tf_nodes.beta, tf_nodes.mean, tf_nodes.variance))
    with open(filepath, 'wb') as f:
        save_layer(node, f, node.name, None, bias)

def save_layer_fcfbn(node, tf_nodes):
    basedir = os.path.dirname(os.path.realpath(__file__))
    filedir = os.path.join(basedir, 'resources/networks/layers')
    filepath = os.path.join(filedir, node.name)
    mkdir_p(filedir)

    weights = tf_nodes.weights.transpose(1, 0)
    bias = np.hstack((tf_nodes.beta, tf_nodes.mean, tf_nodes.variance))
    with open(filepath, 'wb') as f:
        save_layer(node, f, node.name, weights, bias)

        
def save_layer_output(node_name, output):
    print('Saving output %s' % node_name)
    basedir = os.path.dirname(os.path.realpath(__file__))
    filedir = os.path.join(basedir, 'resources/networks/output')
    filepath = os.path.join(filedir, node_name)
    mkdir_p(filedir)

    # NHWC -> NCHW
    if len(output.shape) == 4:
        data = output.transpose((0, 3, 1, 2))
    else:
        data = output
        
    out = FloatArray()
    out.name = node_name
    out.shape.d1, out.shape.d2, out.shape.d3, out.shape.d4 = shape(data)

    for value in data.flatten():
        out.data.append(float(value))

    with open(filepath, 'wb') as f:
        f.write(out.SerializeToString())


def save_layer_maxpool(node, tf_nodes):
    pass


def save_layer_add(node, tf_nodes):
    pass

def save_layer_l2_norm(node, tf_nodes):
    pass

def save_layer_reshape(node, tf_nodes):
    pass

def conv_bias(name, input, prefix):
    return edict({
        'name': name,
        'input': input,
        'weights': '%s/weights' % prefix,
        'conv': '%s/Conv2D' % prefix,
        'bias': '%s/biases' % prefix,
        'output': '%s/BiasAdd' % prefix,
        'type':    'convbias',
    })


def conv_fbn(name, input, prefix):
    return edict({
        'name': name,
        'input': input,
        'output': '%s/Elu' % prefix,
        'conv': '%s/Conv2D' % prefix,
        'weights': '%s/weights' % prefix,
        'beta': '%s/%s/bn/beta' % (prefix, prefix),
        'mean': '%s/%s/bn/moving_mean' % (prefix, prefix),
        'variance': '%s/%s/bn/moving_variance' % (prefix, prefix),
        'fbn': '%s/%s/bn/FusedBatchNorm' % (prefix, prefix),
        'type': 'conv_fbn',        
    })

def fbn(name, input, prefix, i):
    return edict({
        'name': name,
        'input': input,
        'output': 'Elu%s' % ("" if i == 0 else '_%s' % i),
        'beta': '%s/bn/beta' % prefix,
        'mean': '%s/bn/moving_mean' % prefix,
        'variance': '%s/bn/moving_variance' % prefix,
        'fbn': '%s/bn/FusedBatchNorm' % prefix,
        'type': 'fbn',   
    })


def conv1():
    return [
        conv_fbn(name='conv1_1',
                 input='map/TensorArrayStack/TensorArrayGatherV3',
                 prefix='conv1_1'),
        conv_fbn(name='conv1_2',
                 input='conv1_1/Elu',
                 prefix='conv1_2'),
        edict({
            'name': 'pool1',
            'input': 'conv1_2/Elu',
            'output': 'pool1/MaxPool',
            'type': 'maxpool',        
        }),
    ]


def conv2_1():
    return [
        conv_fbn(name='conv2_1_1',
                 input='pool1/MaxPool',
                 prefix='conv2_1/1'),
        conv_bias(name='conv2_1_2',
                  input='conv2_1/1/Elu',
                  prefix='conv2_1/2'),
        edict({
            'name': 'add',
            'input_0': 'pool1/MaxPool',
            'input_1': 'conv2_1/2/BiasAdd',
            'output': 'add',
            'type': 'add',
        }),        
    ]

def conv2_3():
    return [
        fbn(name='conv2_3_bn',
            input='add',
            prefix='conv2_3',
            i=0),
        conv_fbn(name='conv2_3_1',
                 input='Elu',
                 prefix='conv2_3/1'),
        conv_bias(name='conv2_3_2',
                  input='conv2_3/1/Elu',
                  prefix='conv2_3/2'),
        edict({
            'name': 'add_1',
            'input_0': 'add',
            'input_1': 'conv2_3/2/BiasAdd',
            'output': 'add_1',
            'type': 'add',
        }),
    ]

def conv3_1():
    return [
        fbn(name='conv3_1_bn',
            input='add_1',
            prefix='conv3_1',
            i=1),
        conv_fbn(name='conv3_1_1',
                 input='Elu_1',
                 prefix='conv3_1/1'),
        conv_bias(name='conv3_1_2',
                  input='conv3_1/1/Elu',
                  prefix='conv3_1/2'),
        edict({
            'name': 'conv3_1_projection',
            'input': 'add_1',
            'weights': 'conv3_1/projection/weights',
            'output': 'conv3_1/projection/Conv2D',
            'type': 'conv',
        }),
        edict({
            'name': 'add_2',
            'input_0': 'conv3_1/projection/Conv2D',
            'input_1': 'conv3_1/2/BiasAdd',
            'output': 'add_2',
            'type': 'add',
        }),
    ]

def conv3_3():
    return [
        fbn(name='conv3_3_bn',
            input='add_2',
            prefix='conv3_3',
            i=2),
        conv_fbn(name='conv3_3_1',
                 input='Elu_2',
                 prefix='conv3_3/1'),
        conv_bias(name='conv3_3_2',
                  input='conv3_3/1/Elu',
                  prefix='conv3_3/2'),
        edict({
            'name': 'add_3',
            'input_0': 'add_2',
            'input_1': 'conv3_3/2/BiasAdd',
            'output': 'add_3',
            'type': 'add',
        }),
    ]

def conv4_1():
    return [
        fbn(name='conv4_1_bn',
            input='add_3',
            prefix='conv4_1',
            i=3),
        conv_fbn(name='conv4_1_1',
                 input='Elu_3',
                 prefix='conv4_1/1'),
        conv_bias(name='conv4_1_2',
                  input='conv4_1/1/Elu',
                  prefix='conv4_1/2'),
        edict({
            'name': 'conv4_1_projection',
            'input': 'add_3',
            'weights': 'conv4_1/projection/weights',
            'output': 'conv4_1/projection/Conv2D',
            'type': 'conv',
        }),
        edict({
            'name': 'add_2',
            'input_0': 'conv4_1/projection/Conv2D',
            'input_1': 'conv4_1/2/BiasAdd',
            'output': 'add_4',
            'type': 'add',
        }),
    ]

def conv4_3():
    return [
        fbn(name='conv4_3_bn',
            input='add_4',
            prefix='conv4_3',
            i=4),
        conv_fbn(name='conv4_3_1',
                 input='Elu_4',
                 prefix='conv4_3/1'),
        conv_bias(name='conv4_3_2',
                  input='conv4_3/1/Elu',
                  prefix='conv4_3/2'),
        edict({
            'name': 'add_5',
            'input_0': 'add_4',
            'input_1': 'conv4_3/2/BiasAdd',
            'output': 'add_5',
            'type': 'add',
        }),
    ]

def fc_fbn_l2():
    return [
        edict({
            'name': 'flatten_flatten_reshape_add_5',
            'output': 'Flatten/flatten/Reshape',
            'type': 'reshape',
        }),
        edict({
            'name': 'fc1_elu',
            'input': 'Flatten/flatten/Reshape',
            'weights': 'fc1/weights',
            'fc': 'fc1/MatMul',
            'beta': 'fc1/fc1/bn/beta',
            'mean': 'fc1/fc1/bn/moving_mean',
            'variance': 'fc1/fc1/bn/moving_variance',
            'fbn': 'fc1/fc1/bn/FusedBatchNorm',
            'output': 'fc1/Elu',
            'type': 'fcfbn',
        }),
        edict({
            'name': 'ball_reshape_fc1_elu',
            'output': 'ball/Reshape',
            'type': 'reshape',
        }),
        edict({
            'name': 'ball_fbn',
            'input': 'ball/Reshape',
            'output': 'ball/FusedBatchNorm',
            'beta': 'ball/beta',
            'mean': 'ball/moving_mean',
            'variance': 'ball/moving_variance',
            'type': 'fbn',
        }),
        edict({
            'name': 'ball_fbn_reshape',
            'output': 'ball/Reshape_1',
            'type': 'reshape',
        }),
        edict({
            'name': 'fc_fbn_l2',
            'output': 'truediv',
            'type': 'l2_norm',
        }),
    ]   


def definition():
    definition = []

    definition += conv1()
    definition += conv2_1()
    definition += conv2_3()
    definition += conv3_1()
    definition += conv3_3()
    definition += conv4_1()
    definition += conv4_3()
    definition += fc_fbn_l2()

    return definition

def evaluate_node(node, session, feed_dict):
    blacklist = ['name', 'type']
    tf_nodes = edict(dict((k, session.graph.get_tensor_by_name('%s:0' % v))
                          for k, v in node.items() if k not in blacklist))

    return session.run(tf_nodes, feed_dict=feed_dict)
                          

def extract_definition(feed_dict, session):
    input = session.graph.get_tensor_by_name('map/TensorArrayStack/TensorArrayGatherV3:0')
    save_layer_output('input', session.run(input, feed_dict=feed_dict))

    output = session.graph.get_tensor_by_name('truediv:0')
    save_layer_output('output', session.run(output, feed_dict=feed_dict))

    architecture = definition()
    for i, node in enumerate(architecture):
        type = node.type
        tf_nodes = evaluate_node(node, session, feed_dict)

        globals()['save_layer_%s' % type](node, tf_nodes)
        save_layer_output(node.name, tf_nodes.output)

        
def _create_image_encoder(preprocess_fn, factory_fn, image_shape, batch_size=32,
                         session=None, checkpoint_path=None,
                         loss_mode="cosine"):
    image_var = tf.placeholder(tf.uint8, (None, ) + image_shape)

    preprocessed_image_var = tf.map_fn(
        lambda x: preprocess_fn(x, is_training=False),
        tf.cast(image_var, tf.float32))

    l2_normalize = loss_mode == "cosine"
    feature_var, _ = factory_fn(
        preprocessed_image_var, l2_normalize=l2_normalize, reuse=None)
    feature_dim = feature_var.get_shape().as_list()[-1]

    if session is None:
        session = tf.Session()
    if checkpoint_path is not None:
        slim.get_or_create_global_step()
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
            checkpoint_path, slim.get_variables_to_restore())
        session.run(init_assign_op, feed_dict=init_feed_dict)
    
    def encoder(data_x):
        def intercept(x):
            #import pdb; pdb.set_trace()
            extract_definition(x, session)
            return session.run(feature_var, feed_dict=x)
        out = np.zeros((len(data_x), feature_dim), np.float32)
        _run_in_batches(
            intercept,
            {image_var: data_x}, out, batch_size)
        return out

    return encoder


def create_image_encoder(model_filename, batch_size=32, loss_mode="cosine",
                         session=None):
    image_shape = 128, 64, 3
    factory_fn = _network_factory(
        num_classes=1501, is_training=False, weight_decay=1e-8)

    return _create_image_encoder(
        _preprocess, factory_fn, image_shape, batch_size, session,
        model_filename, loss_mode)


def create_box_encoder(model_filename, batch_size=32, loss_mode="cosine"):
    image_shape = 128, 64, 3
    image_encoder = create_image_encoder(model_filename, batch_size, loss_mode)

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches)

    return encoder


def generate_detections(encoder, mot_dir, output_dir, detection_dir=None):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """
    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    for sequence in os.listdir(mot_dir):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        detection_file = os.path.join(
            detection_dir, sequence, "det/det.txt")
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
            bgr_image = cv2.imread(
                image_filenames[frame_idx], cv2.IMREAD_COLOR)
            features = encoder(bgr_image, rows[:, 2:6].copy())
            detections_out += [np.r_[(row, feature)] for row, feature
                               in zip(rows, features)]

        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        np.save(
            output_filename, np.asarray(detections_out), allow_pickle=False)


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.ckpt-68577",
        help="Path to checkpoint file")
    parser.add_argument(
        "--loss_mode", default="cosine", help="Network loss training mode")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--detection_dir", help="Path to custom detections. Defaults to "
        "standard MOT detections Directory structure should be the default "
        "MOTChallenge structure: [sequence]/det/det.txt", default=None)
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
        " exist.", default="detections")
    return parser.parse_args()

def nodes():
    return [n for n in tf.get_default_graph().as_graph_def().node]

if __name__ == "__main__":
    args = parse_args()
    f = create_box_encoder(args.model, batch_size=32, loss_mode=args.loss_mode)
    generate_detections(f, args.mot_dir, args.output_dir, args.detection_dir)
