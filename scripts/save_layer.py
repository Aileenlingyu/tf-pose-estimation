from layer_pb2 import Layer  # NOQA
from float_array_pb2 import FloatArray  # NOQA
import errno

import os, sys
import numpy as np

def shape(arr):
    shape = [1, 1, 1, 1]
    shape[-len(arr.shape):] = arr.shape
    return shape

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

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

def save_layer_fcfbn(node, tf_nodes):
    basedir = os.path.dirname(os.path.realpath(__file__))
    filedir = os.path.join(basedir, 'resources/networks/layers')
    filepath = os.path.join(filedir, node.name)
    mkdir_p(filedir)

    weights = tf_nodes.weights.transpose(1, 0)
    bias = np.hstack((tf_nodes.beta, tf_nodes.mean, tf_nodes.variance))
    with open(filepath, 'wb') as f:
        save_layer(node, f, node.name, weights, bias)

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

