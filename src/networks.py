import os
import tensorflow as tf
from network_mobilenet import MobilenetNetwork
from network_mobilenet_thin import MobilenetNetworkThin
from network_mobilenet_original import MobilenetNetworkOriginal
from network_mobilenet_v2 import MobilenetNetworkV2
from network_vgg16x4 import VGG16x4Network
from network_cmu import CmuNetwork
from resnet32 import Resnet32
from network_mobilenet_fast import MobilenetNetworkFast

base_path = "/home/zaikun/hdd/tf-pose-zaikun/"
def _get_base_path():
    if not os.environ.get('OPENPOSE_MODEL', ''):
        return './models'
    return os.environ.get('OPENPOSE_MODEL')


def get_network(type, placeholder_input, sess_for_load=None, trainable=True):
    if type == 'mobilenet':
        net = MobilenetNetwork({'image': placeholder_input}, conv_width=0.75, conv_width2=1.00, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type == 'mobilenet_fast':
        net = MobilenetNetworkFast({'image': placeholder_input}, conv_width=0.5, conv_width2=0.5, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_0.50_224_2017_06_14/mobilenet_v1_0.50_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type == 'mobilenet_accurate':
        net = MobilenetNetworkThin({'image': placeholder_input}, conv_width=1.00, conv_width2=0.50, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_1.0_224_2017_06_14/mobilenet_v1_1.0_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type == 'mobilenet_thin':
        net = MobilenetNetworkThin({'image': placeholder_input}, conv_width=0.75, conv_width2=0.750, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    
    elif type == 'mobilenet_original':
        net = MobilenetNetworkOriginal({'image': placeholder_input}, conv_width=0.75, conv_width2=0.50, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type == 'mobilenet_v2':
        net = MobilenetNetworkV2({'image': placeholder_input}, conv_width=1, conv_width2=0.50, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v2/model.ckpt-1450000'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type == 'cmu':
        net = CmuNetwork({'image': placeholder_input}, trainable=trainable)
        pretrain_path = 'numpy/openpose_coco.npy'
        last_layer = 'Mconv7_stage6_L{aux}'

    elif type == 'vgg':
        net = CmuNetwork({'image': placeholder_input}, trainable=trainable)
        pretrain_path = 'numpy/openpose_coco.npy'
        last_layer = 'Mconv7_stage6_L{aux}'

    elif type == 'resnet32':
        net = Resnet32({'image': placeholder_input}, conv_width=0.75, conv_width2=0.50, trainable=trainable)
        pretrain_path = 'numpy/resnet32.npy'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type == 'vgg16x4':
        net = VGG16x4Network({'image': placeholder_input}, conv_width=0.75, conv_width2=0.75, trainable=trainable)
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
                'mobilenet': 'models/trained/mobilenet_%s/model-53008' % s,
                'mobilenet_thin' : 'models/trained/mobilenet_thin_432x368/model-160001',
                'mobilenet_original': 'models/trained/mobilenet_thin_benchmark/model-388003',
                'mobilenet_v2': 'models/trained/mobilenet_v2/model-218000',
                'vgg16x4' : 'models/trained/vgg16x4_0.75/model-35000',
            }
            loader = tf.train.Saver()
            loader.restore(sess_for_load, os.path.join(base_path,  ckpts[type]))

    return net, os.path.join(_get_base_path(), pretrain_path), last_layer


def get_graph_path(model_name):
    return {
        'cmu_640x480': os.path.join(base_path,'models/graph/cmu_640x480/graph_opt.pb'),
        'vgg_656x368'        : os.path.join(base_path, 'models/graph/vgg/graph_zaikun_opt.pb'),
        'mobilenet_original_432x368': os.path.join(base_path, './models/graph/mobilenet_thin_432x368/graph_opt.pb'),
        'mobilenet_thin_656x368': os.path.join(base_path, './models/graph/mobilenet_thinzaikun_656x368/graph_opt.pb'),
        'mobilenet_v2_432x368': os.path.join(base_path, './models/graph/mobilenet_v2/graph_freeze.pb'),

    }[model_name]


def model_wh(model_name):
    width, height = model_name.split('_')[-1].split('x')
    return int(width), int(height)

