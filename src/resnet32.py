import tensorflow as tf

import network_base

class Resnet32(network_base.BaseNetwork):
    def __init__(self, inputs, trainable=True, conv_width=1.0, conv_width2=0.75):
        self.conv_width = conv_width
        self.conv_width2 = conv_width2 if conv_width2 else conv_width
        network_base.BaseNetwork.__init__(self, inputs, trainable)
    def setup(self):
        min_depth = 8
        depth2 = lambda d: max(int(d *  self.conv_width2), min_depth)

        (self.feed('image')
         .conv(7, 7, 64, 2, 2, relu=False, name='conv1')
         .batch_normalization(relu=True, name='bn_conv1')
         .max_pool(3, 3, 2, 2, name='pool1')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
         .batch_normalization(name='bn2a_branch1'))

        (self.feed('pool1')
         .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
         .batch_normalization(relu=True, name='bn2a_branch2a')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
         .batch_normalization(relu=True, name='bn2a_branch2b')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
         .batch_normalization(name='bn2a_branch2c'))

        (self.feed('bn2a_branch1',
                   'bn2a_branch2c')
         .add(name='res2a')
         .relu(name='res2a_relu')
         .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
         .batch_normalization(relu=True, name='bn2b_branch2a')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
         .batch_normalization(relu=True, name='bn2b_branch2b')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
         .batch_normalization(name='bn2b_branch2c'))

        (self.feed('res2a_relu',
                   'bn2b_branch2c')
         .add(name='res2b')
         .relu(name='res2b_relu')
         .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
         .batch_normalization(relu=True, name='bn2c_branch2a')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
         .batch_normalization(relu=True, name='bn2c_branch2b')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
         .batch_normalization(name='bn2c_branch2c'))

        (self.feed('res2b_relu',
                   'bn2c_branch2c')
         .add(name='res2c')
         .relu(name='res2c_relu')
         .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1')
         .batch_normalization(name='bn3a_branch1'))

        (self.feed('res2c_relu')
         .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
         .batch_normalization(relu=True, name='bn3a_branch2a')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
         .batch_normalization(relu=True, name='bn3a_branch2b')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
         .batch_normalization(name='bn3a_branch2c'))

        (self.feed('bn3a_branch1',
                   'bn3a_branch2c')
         .add(name='res3a')
         .relu(name='res3a_relu')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b_branch2a')
         .batch_normalization(relu=True, name='bn3b_branch2a')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b_branch2b')
         .batch_normalization(relu=True, name='bn3b_branch2b')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b_branch2c')
         .batch_normalization(name='bn3b_branch2c'))

        (self.feed('res3a_relu',
                   'bn3b_branch2c')
         .add(name='res3b')
         .relu(name='res3b_relu')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3c_branch2a')
         .batch_normalization(relu=True, name='bn3c_branch2a')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3c_branch2b')
         .batch_normalization(relu=True, name='bn3c_branch2b')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3c_branch2c')
         .batch_normalization(name='bn3c_branch2c'))

        (self.feed('res3b_relu',
                   'bn3c_branch2c')
         .add(name='res3c')
         .relu(name='res3c_relu')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3d_branch2a')
         .batch_normalization(relu=True, name='bn3d_branch2a')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3d_branch2b')
         .batch_normalization(relu=True, name='bn3d_branch2b')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3d_branch2c')
         .batch_normalization(name='bn3d_branch2c'))

        (self.feed('res3c_relu',
                   'bn3d_branch2c')
         .add(name='res3d')
         .relu(name='res3d_relu')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch1')
         .batch_normalization(name='bn4a_branch1'))

        (self.feed('res3d_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4a_branch2a')
         .batch_normalization(relu=True, name='bn4a_branch2a')
         .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4a_branch2b')
         .batch_normalization(relu=True, name='bn4a_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
         .batch_normalization(name='bn4a_branch2c'))

        (self.feed('bn4a_branch1',
                   'bn4a_branch2c')
         .add(name='res4a')
         .relu(name='res4a_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b_branch2a')
         .batch_normalization(relu=True, name='bn4b_branch2a')
         .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b_branch2b')
         .batch_normalization(relu=True, name='bn4b_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b_branch2c')
         .batch_normalization(name='bn4b_branch2c'))

        (self.feed('res4a_relu',
                   'bn4b_branch2c')
         .add(name='res4b')
         .relu(name='res4b_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4c_branch2a')
         .batch_normalization(relu=True, name='bn4c_branch2a'))

        feature_lv = 'bn4c_branch2a'
        prefix = 'MConv_Stage1'
        (self.feed(feature_lv)
         .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_1')
         .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_2')
         .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_3')
         .separable_conv(1, 1, depth2(512), 1, name=prefix + '_L1_4')
         .separable_conv(1, 1, 38, 1, relu=False, name=prefix + '_L1_5'))

        (self.feed(feature_lv)
         .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_1')
         .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_2')
         .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_3')
         .separable_conv(1, 1, depth2(512), 1, name=prefix + '_L2_4')
         .separable_conv(1, 1, 19, 1, relu=False, name=prefix + '_L2_5'))

        for stage_id in range(5):
            prefix_prev = 'MConv_Stage%d' % (stage_id + 1)
            prefix = 'MConv_Stage%d' % (stage_id + 2)
            (self.feed(prefix_prev + '_L1_5',
                       prefix_prev + '_L2_5',
                       feature_lv)
             .concat(3, name=prefix + '_concat')
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_1')
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_2')
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_3')
             .separable_conv(1, 1, depth2(128), 1, name=prefix + '_L1_4')
             .separable_conv(1, 1, 38, 1, relu=False, name=prefix + '_L1_5'))
            (self.feed(prefix + '_concat')
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_1')
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_2')
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_3')
             .separable_conv(1, 1, depth2(128), 1, name=prefix + '_L2_4')
             .separable_conv(1, 1, 19, 1, relu=False, name=prefix + '_L2_5'))

        # final result
        (self.feed('MConv_Stage6_L2_5',
                   'MConv_Stage6_L1_5')
         .concat(3, name='concat_stage7'))

    def loss_l1_l2(self):
        l1s = []
        l2s = []
        for layer_name in sorted(self.layers.keys()):
            if '_L1_5' in layer_name:
                l1s.append(self.layers[layer_name])
            if '_L2_5' in layer_name:
                l2s.append(self.layers[layer_name])

        return l1s, l2s

    def loss_last(self):
        return self.get_output('MConv_Stage6_L1_5'), self.get_output('MConv_Stage6_L2_5')

    def restorable_variables(self):
        return None