import tensorflow as tf

import network_base


class MobilenetNetworkThinFatBranch(network_base.BaseNetwork):
    def __init__(self, inputs, trainable=True, conv_width=1.0, conv_width2=None):
        self.conv_width = conv_width
        self.conv_width2 = conv_width2 if conv_width2 else conv_width
        network_base.BaseNetwork.__init__(self, inputs, trainable)

    def setup(self):
        min_depth = 8
        depth = lambda d: max(int(d * self.conv_width), min_depth)
        depth2 = lambda d: max(int(d * self.conv_width2), min_depth)

        with tf.variable_scope(None, 'MobilenetV1'):
            (self.feed('image')
             .convb(3, 3, depth(32), 2, name='Conv2d_0')
             .separable_conv(3, 3, depth(64), 1, name='Conv2d_1')
             .separable_conv(3, 3, depth(128), 2, name='Conv2d_2')
             .separable_conv(3, 3, depth(128), 1, name='Conv2d_3')
             .separable_conv(3, 3, depth(256), 2, name='Conv2d_4')
             .separable_conv(3, 3, depth(256), 1, name='Conv2d_5')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_6')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_7')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_8')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_9')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_10')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_11')
             # .separable_conv(3, 3, depth(1024), 2, name='Conv2d_12')
             # .separable_conv(3, 3, depth(1024), 1, name='Conv2d_13')
             )

        (self.feed('Conv2d_3').max_pool(2, 2, 2, 2, name='Conv2d_3_pool'))

        (self.feed('Conv2d_3_pool', 'Conv2d_7', 'Conv2d_11')
            .concat(3, name='feat_concat'))

        feature_lv = 'feat_concat'
        (self.feed(feature_lv)
        .conv(1, 3, depth2(256), 1, 1, name='conv4_3_CPM')
        .conv(1, 1, depth2(128), 1, 1, name='conv4_4_CPM')
        .conv(3, 3, depth2(128), 1, 1, name='conv5_1_CPM_L1')
        .conv(3, 3, depth2(128), 1, 1, name='conv5_2_CPM_L1')
        .conv(3, 3, depth2(128), 1, 1, name='conv5_3_CPM_L1')
        .conv(1, 1, depth2(512), 1, 1, name='conv5_4_CPM_L1')
        .conv(1, 1, 38, 1, 1, relu=False, name='conv5_5_CPM_L1'))

        (self.feed('feat_concat')
        .conv(3, 3, depth2(128), 1, 1, name='conv5_1_CPM_L2')
        .conv(3, 3, depth2(128), 1, 1, name='conv5_2_CPM_L2')
        .conv(3, 3, depth2(128), 1, 1, name='conv5_3_CPM_L2')
        .conv(1, 1, depth2(512), 1, 1, name='conv5_4_CPM_L2')
        .conv(1, 1, 19, 1, 1, relu=False, name='conv5_5_CPM_L2'))

        (self.feed('conv5_5_CPM_L1',
                   'conv5_5_CPM_L2',
                   'feat_concat')
        .concat(3, name='concat_stage2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv1_stage2_L1')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv2_stage2_L1')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv3_stage2_L1')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv4_stage2_L1')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv5_stage2_L1')
        .conv(1, 1, depth2(128), 1, 1, name='Mconv6_stage2_L1')
        .conv(1, 1, 38, 1, 1, relu=False, name='Mconv7_stage2_L1'))

        (self.feed('concat_stage2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv1_stage2_L2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv2_stage2_L2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv3_stage2_L2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv4_stage2_L2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv5_stage2_L2')
        .conv(1, 1, depth2(128), 1, 1, name='Mconv6_stage2_L2')
        .conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage2_L2'))

        (self.feed('Mconv7_stage2_L1',
                   'Mconv7_stage2_L2',
                   'feat_concat')
        .concat(3, name='concat_stage3')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv1_stage3_L1')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv2_stage3_L1')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv3_stage3_L1')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv4_stage3_L1')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv5_stage3_L1')
        .conv(1, 1, depth2(128), 1, 1, name='Mconv6_stage3_L1')
        .conv(1, 1, 38, 1, 1, relu=False, name='Mconv7_stage3_L1'))

        (self.feed('concat_stage3')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv1_stage3_L2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv2_stage3_L2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv3_stage3_L2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv4_stage3_L2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv5_stage3_L2')
        .conv(1, 1, depth2(128), 1, 1, name='Mconv6_stage3_L2')
        .conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage3_L2'))

        (self.feed('Mconv7_stage3_L1',
                   'Mconv7_stage3_L2',
                   'feat_concat')
        .concat(3, name='concat_stage4')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv1_stage4_L1')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv2_stage4_L1')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv3_stage4_L1')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv4_stage4_L1')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv5_stage4_L1')
        .conv(1, 1, depth2(128), 1, 1, name='Mconv6_stage4_L1')
        .conv(1, 1, 38, 1, 1, relu=False, name='Mconv7_stage4_L1'))

        (self.feed('concat_stage4')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv1_stage4_L2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv2_stage4_L2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv3_stage4_L2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv4_stage4_L2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv5_stage4_L2')
        .conv(1, 1, depth2(128), 1, 1, name='Mconv6_stage4_L2')
        .conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage4_L2'))

        (self.feed('Mconv7_stage4_L1',
                   'Mconv7_stage4_L2',
                   'feat_concat')
        .concat(3, name='concat_stage5')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv1_stage5_L1')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv2_stage5_L1')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv3_stage5_L1')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv4_stage5_L1')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv5_stage5_L1')
        .conv(1, 1, depth2(128), 1, 1, name='Mconv6_stage5_L1')
        .conv(1, 1, 38, 1, 1, relu=False, name='Mconv7_stage5_L1'))

        (self.feed('concat_stage5')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv1_stage5_L2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv2_stage5_L2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv3_stage5_L2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv4_stage5_L2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv5_stage5_L2')
        .conv(1, 1, depth2(128), 1, 1, name='Mconv6_stage5_L2')
        .conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage5_L2'))

        (self.feed('Mconv7_stage5_L1',
                   'Mconv7_stage5_L2',
                   'feat_concat')
        .concat(3, name='concat_stage6')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv1_stage6_L1')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv2_stage6_L1')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv3_stage6_L1')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv4_stage6_L1')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv5_stage6_L1')
        .conv(1, 1, depth2(128), 1, 1, name='Mconv6_stage6_L1')
        .conv(1, 1, 38, 1, 1, relu=False, name='Mconv7_stage6_L1'))

        (self.feed('concat_stage6')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv1_stage6_L2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv2_stage6_L2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv3_stage6_L2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv4_stage6_L2')
        .conv(7, 7, depth2(128), 1, 1, name='Mconv5_stage6_L2')
        .conv(1, 1, depth2(128), 1, 1, name='Mconv6_stage6_L2')
        .conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage6_L2'))

        with tf.variable_scope('Openpose'):
            (self.feed('Mconv7_stage6_L2',
                       'Mconv7_stage6_L1')
        .concat(3, name='concat_stage7'))


    def loss_l1_l2(self):
         l1s = []
         l2s = []
         for layer_name in self.layers.keys():
              if 'Mconv7' in layer_name and '_L1' in layer_name:
                   l1s.append(self.layers[layer_name])
              if 'Mconv7' in layer_name and '_L2' in layer_name:
                   l2s.append(self.layers[layer_name])

         return l1s, l2s

    def loss_last(self):
         return self.get_output('Mconv7_stage6_L1'), self.get_output('Mconv7_stage6_L2')

    def restorable_variables(self):
        vs = {v.op.name: v for v in tf.global_variables() if
              'MobilenetV1/Conv2d' in v.op.name and
              # 'global_step' not in v.op.name and
              # 'beta1_power' not in v.op.name and 'beta2_power' not in v.op.name and
              'RMSProp' not in v.op.name and 'Momentum' not in v.op.name and
              'Ada' not in v.op.name and 'Adam' not in v.op.name
              }
        return vs
