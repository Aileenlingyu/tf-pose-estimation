# vim: expandtab:ts=4:sw=4
import pickle
import os
import sys
import errno
import argparse
import numpy as np
import cv2
from save_layer import *
from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
from easydict import EasyDict as edict
sys.path.append('./src/proto')
sys.path.append('./src')
sys.path.append('./script')
from networks import get_network

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True


def save_layer_maxpool(node, tf_nodes):
    pass


def save_layer_add(node, tf_nodes):
    pass

def save_layer_concat(node, tf_nodes):
    pass

def save_layer_l2_norm(node, tf_nodes):
    pass

def save_layer_reshape(node, tf_nodes):
    pass


def save_layer_depthwise(node, tf_nodes):
    # save weights
    basedir = os.path.dirname(os.path.realpath(__file__))
    filedir = os.path.join(basedir, 'resources/networks/layers')
    filepath = os.path.join(filedir, node.name.split('/')[-1] + '_depthwise')
    mkdir_p(filedir)

    weights = tf_nodes.weights.transpose((3, 2, 0, 1))
    with open(filepath, 'wb') as f:
        save_layer(node, f, node.name, weights, None)


def save_layer_fbn(node, tf_nodes):
    basedir = os.path.dirname(os.path.realpath(__file__))
    filedir = os.path.join(basedir, 'resources/networks/layers')
    filepath = os.path.join(filedir, node.name.split('/')[-1] + '_pointwise')
    mkdir_p(filedir)

    weights = tf_nodes.weights.transpose(3, 2, 0, 1)
    bias = np.hstack((tf_nodes.beta, tf_nodes.mean, tf_nodes.variance))
    with open(filepath, 'wb') as f:
        save_layer(node, f, node.name, weights, bias)

def save_layer_fbn_norelu(node, tf_nodes):
    basedir = os.path.dirname(os.path.realpath(__file__))
    filedir = os.path.join(basedir, 'resources/networks/layers')
    filepath = os.path.join(filedir, node.name.split('/')[-1] + '_pointwise')
    mkdir_p(filedir)

    weights = tf_nodes.weights.transpose(3, 2, 0, 1)
    bias = np.hstack((tf_nodes.beta, tf_nodes.mean, tf_nodes.variance))
    with open(filepath, 'wb') as f:
        save_layer(node, f, node.name, weights, bias)



def depthwise(name, input, prefix):
    return edict({
        'name': name,
        'input': input,
        'output': '%s/depthwise' % prefix,
        'depthwise': '%s/depthwise' % prefix,
        'weights': '%s/depthwise_weights' % prefix,
        'type': 'depthwise',
    })

def conv_fbn(name, input, prefix):
    return edict({
        'name': name,
        'input': input,
        'output': '%s/Relu' % prefix,
        'conv': '%s/Conv2D' % prefix,
        'weights': '%s/weights' % prefix,
        'beta': '%s/BatchNorm/beta' % (prefix),
        'mean': '%s/BatchNorm/moving_mean' % (prefix),
        'variance': '%s/BatchNorm/moving_variance' % (prefix),
        'fbn': '%s/BatchNorm/FusedBatchNorm' % (prefix),
        'type': 'fbn'
    })

def separable_conv(name, input, prefix, use_relu):
    return [
        depthwise(name, input, prefix + '_depthwise'),
        conv_fbn(name,  prefix + '_depthwise/depthwise', prefix + '_pointwise') if use_relu else conv_fbn_norelu(name, input, prefix + '_pointwise')
    ]


def conv_fbn_norelu(name, input, prefix):
    return edict({
        'name': name,
        'input': input,
        'conv': '%s/Conv2D' % prefix,
        'weights': '%s/weights' % prefix,
        'beta': '%s/BatchNorm/beta' % (prefix),
        'mean': '%s/BatchNorm/moving_mean' % (prefix),
        'variance': '%s/BatchNorm/moving_variance' % (prefix),
        'fbn': '%s/BatchNorm/FusedBatchNorm' % (prefix),
        'output' : '%s/BatchNorm/FusedBatchNorm' % prefix,
        'type': 'fbn_norelu'
    })

def conv1():
    return conv_fbn(name='Conv2d_0',
                 input='TfPoseEstimator/image',
                 prefix='TfPoseEstimator/MobilenetV1/Conv2d_0')


def conv2d_3_pool():
    return [
        edict({
        'name': 'conv2d_3_pool',
        'input': 'TfPoseEstimator/MobilenetV1/Conv2d_3_pointwise/Relu',
        'output': 'TfPoseEstimator/Conv2d_3_pool',
        'type': 'maxpool'
    })]

def concat():
    return [
        edict({
            'name': 'feat_concat',
            'input_0': 'TfPoseEstimator/Conv2d_3_pool',
            'input_1': 'TfPoseEstimator/MobilenetV1/Conv2d_7_pointwise/Relu',
            'input_2': 'TfPoseEstimator/MobilenetV1/Conv2d_11_pointwise/Relu',
            'output': 'TfPoseEstimator/feat_concat',
            'type': 'concat'
        })
    ]

def backbone():
    arch = [ conv1()]
    for i in range(1,12):
        name = 'TfPoseEstimator/MobilenetV1/Conv2d_%s' % i
        prefix = name
        if i == 1:
            input = 'TfPoseEstimator/MobilenetV1/Conv2d_0/Relu'
        else:
            input = 'TfPoseEstimator/MobilenetV1/Conv2d_%s_pointwise/Relu' % (i-1)
        arch.extend(separable_conv(name, input, prefix, True))

    return arch


def stages(stage_n):
    arch = []
    for i in range(5):
        for Lstage in range(1,3):
            name = 'TfPoseEstimator/Openpose/MConv_Stage%s_L%s_%s' % (stage_n,Lstage, i+1)
            if i == 0:
                prefix = name
                input = 'TfPoseEstimator/feat_concat'
            else:
                prefix = name
                input = 'TfPoseEstimator/Openpose/MConv_Stage%s_L%s_%s_pointwise/Relu' %(stage_n, Lstage, i)
            if i != 4:
                arch.extend(separable_conv(name, input, prefix, True))
            else:
                arch.extend(separable_conv(name, input, prefix, False))
    return arch

def define_networks():
    nets = []
    nets = nets + backbone()
    nets += conv2d_3_pool()
    nets += concat()
    for s in range(1,7):
        nets = nets + stages(s)

    return nets

def evaluate_node(node, session, feed_dict):
    blacklist = ['name', 'type']
    tf_nodes = edict(dict((k, session.graph.get_tensor_by_name('%s:0' % v))
                          for k, v in node.items() if k not in blacklist))

    return session.run(tf_nodes, feed_dict=feed_dict)

def run_all(graph_path):
    # load graph
    with tf.gfile.GFile(graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    graph = tf.get_default_graph()
    tf.import_graph_def(graph_def, name='TfPoseEstimator')
    sess = tf.Session(graph=graph)

    for op in graph.get_operations():
        print(op.name)
    #import pdb; pdb.set_trace()
    sess.graph.get_tensor_by_name('TfPoseEstimator/MobilenetV1/Conv2d_0/weights:0')
    architecture = define_networks()
    image_shape = 368, 432, 3
    patch = np.random.uniform(
        0., 255., image_shape).astype(np.uint8)

    feed_dict = [ patch ]
    for i, node in enumerate(architecture):
        #import pdb; pdb.set_trace()
        type = node.type
        tf_nodes = evaluate_node(node, sess, {'TfPoseEstimator/image:0': feed_dict} )

        globals()['save_layer_%s' % type](node, tf_nodes)
        #save_layer_output(node.name, tf_nodes.output)

class convert_mobilepose(object):
    def __init__(self, imgpath, graph_path, height, width):
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())
        self.feed = np.array(Image.open(imgpath).resize( (height, width), Image.ANTIALIAS))[np.newaxis,:]
        self.graph_path = graph_path

    def get_graph(self, graph_path):
        with tf.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        graph = tf.get_default_graph()
        tf.import_graph_def(graph_def, name='TfPoseEstimator')
        persistent_sess = tf.Session(graph=graph)

        for op in graph.get_operations():
            print(op.name)
        #import pdb; pdb.set_trace()
        return graph

    def save_bn_dw(self, prefix_output, prefix, postfix, has_relu=True):
        depthwise = "{}conv2d_{}_depthwise_weights".format(prefix_output, postfix)
        t = self.session.graph.get_tensor_by_name("{}_{}/depthwise_weights:0".format(prefix, postfix))
        save_layer_output(depthwise, self.session.run(t, feed_dict={'TfPoseEstimator/image:0': self.feed}))


    def save_bn_pw(self, prefix_output, prefix, postfix, has_relu=True):
        const = '{}conv2d_{}_const'.format(prefix_output, postfix)
        const_tensor = self.session.graph.get_tensor_by_name("{}_{}/BatchNorm/Const:0".format(prefix, postfix))
        save_layer_output(const, self.session.run(const_tensor, feed_dict= {'TfPoseEstimator/image:0' : self.feed}))

        beta = '{}conv2d_{}_beta'.format(prefix_output, postfix)
        beta_tensor = self.session.graph.get_tensor_by_name("{}_{}/BatchNorm/beta:0".format(prefix, postfix))
        save_layer_output(beta, self.session.run(beta_tensor, feed_dict= {'TfPoseEstimator/image:0' : self.feed}))

        mean = '{}conv2d_{}_mean'.format(prefix_output, postfix)
        mean_tensor  = self.session.graph.get_tensor_by_name("{}_{}/BatchNorm/moving_mean/read:0".format(prefix, postfix))
        save_layer_output(mean, self.session.run(mean_tensor, feed_dict= {'TfPoseEstimator/image:0' : self.feed}))

        var = '{}conv2d_{}_var'.format(prefix_output, postfix)
        var_tensor  = self.session.graph.get_tensor_by_name("{}_{}/BatchNorm/moving_variance/read:0".format(prefix, postfix))
        save_layer_output(var, self.session.run(var_tensor, feed_dict= {'TfPoseEstimator/image:0' : self.feed}))

        # conv0_fbn   = self.session.graph.get_tensor_by_name("TfPoseEstimator/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNorm:0")
        # save_layer_output('conv2d_0_fbn', self.session.run(conv0_fbn, feed_dict= {'TfPoseEstimator/image:0' : self.feed}))

        if (has_relu):
            relu = '{}conv2d_{}_relu'.format(prefix_output, postfix)
            relu_tensor = self.session.graph.get_tensor_by_name("{}_{}/Relu:0".format(prefix, postfix))
            save_layer_output(relu, self.session.run(relu_tensor, feed_dict = {'TfPoseEstimator/image:0' : self.feed}))

    def convert_conv0(self):
        #slim.get_or_create_global_step()
        # init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
        #     model_path, slim.get_variables_to_restore())
        # session.run(init_assign_op, feed_dict=init_feed_dict)
        # imported_meta = tf.train.import_meta_graph(model_path + ".meta")
        # imported_meta.restore(session, model_path)
        graph = self.get_graph(self.graph_path)
        input = self.session.graph.get_tensor_by_name('TfPoseEstimator/image:0')
        save_layer_output("input", self.session.run(input, feed_dict = {'TfPoseEstimator/image:0' : self.feed}))
        #tensor_output = session.graph.get_tensor_by_name('TfPoseEstimator/Openpose/concat_stage7:0')
        #import pdb; pdb.set_trace()

        print('lodaing tensor by name')
        #prefix = 'conv0_'
        conv0_weights = self.session.graph.get_tensor_by_name('TfPoseEstimator/MobilenetV1/Conv2d_0/weights:0')
        save_layer_output('conv2d_0_weights', self.session.run(conv0_weights, feed_dict= {'TfPoseEstimator/image:0' : self.feed}))

        #conv0_conv2d = self.session.graph.get_tensor_by_name('TfPoseEstimator/MobilenetV1/Conv2d_0/Conv2D:0')
        #save_layer_output('conv2d_0_conv2d', self.session.run(conv0_conv2d, feed_dict= {'TfPoseEstimator/image:0' : self.feed}))

        self.save_bn_pw('', 'TfPoseEstimator/MobilenetV1/Conv2d' , 0 , True)

    def mobilepose_backbone(self):
        # conv2d_1 to conv2d_11
        for i in range(1, 12):
            dw = self.session.graph.get_tensor_by_name("TfPoseEstimator/MobilenetV1/Conv2d_{}_depthwise/depthwise_weights:0".format(i))
            #dw_out = self.session.graph.get_tensor_by_name("TfPoseEstimator/MobilenetV1/Conv2d_{}_depthwise/depthwise:0".format(i))
            pw =  self.session.graph.get_tensor_by_name("TfPoseEstimator/MobilenetV1/Conv2d_{}_pointwise/weights:0".format(i))
            #pw_out = self.session.graph.get_tensor_by_name("TfPoseEstimator/MobilenetV1/Conv2d_{}_pointwise/Conv2D:0".format(i))

            # const = self.session.graph.get_tensor_by_name('TfPoseEstimator/MobilenetV1/Conv2d_{}_pointwise/BatchNorm/Const:0'.format(i))
            # beta  = self.session.graph.get_tensor_by_name('TfPoseEstimator/MobilenetV1/Conv2d_{}_pointwise/BatchNorm/beta/read:0'.format(i))
            # mean  = self.session.graph.get_tensor_by_name('TfPoseEstimator/MobilenetV1/Conv2d_{}_pointwise/BatchNorm/moving_mean/read:0'.format(i))
            # var   = self.session.graph.get_tensor_by_name('TfPoseEstimator/MobilenetV1/Conv2d_{}_pointwise/BatchNorm/moving_variance/read:0'.format(i))
            # fbn = self.session.graph.get_tensor_by_name('TfPoseEstimator/MobilenetV1/Conv2d_{}_pointwise/BatchNorm/FusedBatchNorm:0'.format(i))
            #
            # relu = self.session.graph.get_tensor_by_name("TfPoseEstimator/MobilenetV1/Conv2d_{}_pointwise/Relu:0".format(i))
            # print(session.run(dw_out, feed_dict = {'TfPoseEstimator/image:0' : feed}).shape)
            # print(session.run(pw_out, feed_dict = {'TfPoseEstimator/image:0' : feed}).shape)
            # print(session.run(bn_out, feed_dict = {'TfPoseEstimator/image:0' : feed}).shape)
            self.save_bn_dw('', 'TfPoseEstimator/MobilenetV1/Conv2d', "{}_depthwise".format(i), True)
            self.save_bn_pw('', 'TfPoseEstimator/MobilenetV1/Conv2d', "{}_pointwise".format(i), True)


    def mobilepose_branch(self, branch):
        # conv2d_3_pool
        con2d_3_pool = self.session.graph.get_tensor_by_name("TfPoseEstimator/Conv2d_3_pool:0")

        '''
            name: "TfPoseEstimator/feat_concat"
            op: "ConcatV2"
            input: "TfPoseEstimator/Conv2d_3_pool"
            input: "TfPoseEstimator/MobilenetV1/Conv2d_7_pointwise/Relu"
            input: "TfPoseEstimator/MobilenetV1/Conv2d_11_pointwise/Relu"
            input: "TfPoseEstimator/feat_concat/axis"
        '''
        # feat_concat
        feat_concat = self.session.graph.get_tensor_by_name("TfPoseEstimator/feat_concat:0")

        for stage in range(1, 7):
            for layer_number in range(1, 6):
                curr_prefix = 'TfPoseEstimator/Openpose/MConv_Stage{}_{}'.format(stage, branch)
                dw = self.session.graph.get_tensor_by_name("TfPoseEstimator/Openpose/MConv_Stage{}_{}_{}_depthwise/depthwise_weights:0".format(stage, branch, layer_number))
                #dw_out = self.session.graph.get_tensor_by_name("TfPoseEstimator/Openpose/MConv_Stage{}_{}_{}_depthwise/depthwise:0".format(stage, branch, layer_number))
                pw = self.session.graph.get_tensor_by_name("TfPoseEstimator/Openpose/MConv_Stage{}_{}_{}_pointwise/weights:0".format(stage, branch, layer_number))
                #pw_out = self.session.graph.get_tensor_by_name("TfPoseEstimator/Openpose/MConv_Stage{}_{}_{}_pointwise/Conv2D:0".format(stage, branch, layer_number))

                # const = self.session.graph.get_tensor_by_name(curr_prefix + "_{}_pointwise/BatchNorm/Const:0".format(stage, branch, layer_number))
                # beta = self.session.graph.get_tensor_by_name("TfPoseEstimator/Openpose/MConv_Stage{}_{}_{}_pointwise/BatchNorm/beta/read:0".format(stage, branch, layer_number))
                # mean = self.session.graph.get_tensor_by_name("TfPoseEstimator/Openpose/MConv_Stage{}_{}_{}_pointwise/BatchNorm/moving_mean/read:0".format(stage, branch, layer_number))
                # var = self.session.graph.get_tensor_by_name("TfPoseEstimator/Openpose/MConv_Stage{}_{}_{}_pointwise/BatchNorm/moving_variance/read:0".format(stage, branch, layer_number))
                # fbn = self.session.graph.get_tensor_by_name("TfPoseEstimator/Openpose/MConv_Stage{}_{}_{}_pointwise/BatchNorm/FusedBatchNorm:0".format(stage, branch, layer_number))
                #
                # if layer_number < 5 :
                #     relu = self.session.graph.get_tensor_by_name("TfPoseEstimator/Openpose/MConv_Stage{}_{}_{}_pointwise/Relu:0".format(stage, branch, layer_number))
                self.save_bn_dw('stage{}_branch{}_'.format(stage, branch), curr_prefix, '{}_depthwise'.format(layer_number), False)
                self.save_bn_pw('stage{}_branch{}_'.format(stage, branch), curr_prefix, '{}_pointwise'.format(layer_number), layer_number < 5)

    def nodes(self):
        return [n for n in tf.get_default_graph().as_graph_def().node]
        #epsilon = 0.0010000000474974513

if __name__ == '__main__':
    # print("using freezed graph model for conversion")
    #model = convert_mobilepose('4kids.jpg', './models/graph/mobilenet_thinzaikun_432x368/graph_freeze.pb', 432, 368)
    #model.convert_conv0()
    # model.mobilepose_backbone()
    # model.mobilepose_branch('L1')
    # model.mobilepose_branch('L2')
    #run_all('../../../../tf-pose-estimation/models/graph/mobilenet_thin/graph_opt.pb')
    run_all('/home/zaikun/hdd/tec2/technology/convert_model/tf_cudnn/model/graph_freeze.pb')
