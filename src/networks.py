import os

import tensorflow as tf
from network_mobilenet import MobilenetNetwork
from network_mobilenet_thin import MobilenetNetworkThin
from network_vgg16x4 import VGG16x4Network
from network_cmu import CmuNetwork
from network_mobilenet_v2_tf import MobilenetNetworkV2
from network_mobilenet_v2_all import MobilenetNetworkV2All
from resnet32 import Resnet32

def _get_base_path():
    if not os.environ.get('OPENPOSE_MODEL', ''):
        return './models'
    return os.environ.get('OPENPOSE_MODEL')


def get_network(type, placeholder_input, sess_for_load=None, trainable=True):
    if type == 'mobilenet':
        net = MobilenetNetwork({'image': placeholder_input}, conv_width=0.75, conv_width2=1.00, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt'
        last_layer = 'MConv_Stage5_L{aux}_5'
    elif type == 'mobilenet_fast':
        net = MobilenetNetworkThin({'image': placeholder_input}, conv_width=0.5, conv_width2=0.5, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_0.50_224_2017_06_14/mobilenet_v1_0.50_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type == 'mobilenet_accurate':
        net = MobilenetNetwork({'image': placeholder_input}, conv_width=1.00, conv_width2=0.50, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_1.0_224_2017_06_14/mobilenet_v1_1.0_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type == 'mobilenet_thin':
        net = MobilenetNetworkThin({'image': placeholder_input}, conv_width=0.75, conv_width2=0.50, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type == 'mobilenet_v2':
        net = MobilenetNetworkV2All({'image': placeholder_input}, conv_width=1, conv_width2=0.50, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2/model.ckpt-1450000'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type == 'cmu':
        net = CmuNetwork({'image': placeholder_input}, trainable=trainable)
        pretrain_path = 'numpy/openpose_coco.npy'
        se = 'Mconv7_stage6_L{aux}'
    elif type == 'vgg':
        net = CmuNetwork({'image': placeholder_input}, trainable=trainable)
        pretrain_path = 'numpy/openpose_vgg16.npy'
        last_layer = 'Mconv7_stage6_L{aux}'

    elif type == 'resnet32':
        net = Resnet32({'image': placeholder_input}, conv_width=0.75, conv_width2=0.50, trainable=trainable)
        pretrain_path = 'numpy/resnet32.npy'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type == 'vgg16x4':
        net = VGG16x4Network({'image': placeholder_input}, trainable=trainable)
        pretrain_path = 'numpy/vgg16x4.npy'
        last_layer = 'Mconv7_stage6_L{aux}'
    else:
        raise Exception('Invalid Mode.')

    if sess_for_load is not None:
        if type == 'cmu' or type == 'vgg':
            net.load(os.path.join(_get_base_path(), pretrain_path), sess_for_load)
        else:
            s = '%dx%d' % (placeholder_input.shape[2], placeholder_input.shape[1])
            ckpts = {
                'mobilenet': 'trained/mobilenet_%s/model-53008' % s,
                'mobilenet_thin': 'pretrained/mobilenet_0.75_0.50_model-388003/model-388003',
                'mobilenet_fast': 'trained/mobilenet_fast_%s/model-189000' % s,
                'mobilenet_accurate': 'trained/mobilenet_accurate/model-170000'
            }
            loader = tf.train.Saver()
            loader.restore(sess_for_load, os.path.join(_get_base_path(), ckpts[type]))

    return net, os.path.join(_get_base_path(), pretrain_path), last_layer


def get_graph_path(model_name):
    return {
        'cmu_640x480': './models/graph/cmu_640x480/graph_opt.pb',
        'cmuq_640x480': './models/graph/cmu_640x480/graph_q.pb',

        'cmu_640x360': './models/graph/cmu_640x360/graph_opt.pb',
        'cmuq_640x360': './models/graph/cmu_640x360/graph_q.pb',
        'mobilenet_thinwide_368x368': './model/mobilenet_thin_wide/graph_opt_mobilenet_26000.pb',
        'mobilenet_thin_432x368': './models/graph/mobilenet_thin_432x368/graph_opt.pb',
        'mobilenet_368x368': './model/mobilenet_batch:16/graph_opt_mobilenet_35005_batch_16.pb',

    }[model_name]


def model_wh(model_name):
    width, height = model_name.split('_')[-1].split('x')
    return int(width), int(height)
