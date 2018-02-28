import tensorflow as tf
import network_base

class MobilenetV2(network_base.BaseNetwork):
    def __init__(self, inputs, trainable=True, conv_width=1.0, conv_width2=None):
        self.conv_width = conv_width
        self.conv_width2 = conv_width2 if conv_width2 else conv_width
        network_base.BaseNetwork.__init__(self, inputs, trainable)

    def setup(self):
        min_depth = 8
        depth = lambda d: max(int(d * self.conv_width), min_depth)
        depth2 = lambda d: max(int(d * self.conv_width2), min_depth)

        def setup(self):
            (self.feed('image')
             .conv(3, 3, 32, 2, 2, biased=False, relu=False, name='conv1')
             .batch_normalization(relu=True, name='conv1_bn')
             .conv(1, 1, 32, 1, 1, biased=False, relu=False, name='conv2_1_expand')
             .batch_normalization(relu=True, name='conv2_1_expand_bn')
             .conv(3, 3, 32, 1, 1, biased=False, group=32, relu=False, name='conv2_1_dwise')
             .batch_normalization(relu=True, name='conv2_1_dwise_bn')
             .conv(1, 1, 16, 1, 1, biased=False, relu=False, name='conv2_1_linear')
             .batch_normalization(name='conv2_1_linear_bn')
             .conv(1, 1, 96, 1, 1, biased=False, relu=False, name='conv2_2_expand')
             .batch_normalization(relu=True, name='conv2_2_expand_bn')
             .conv(3, 3, 96, 2, 2, biased=False, group=96, relu=False, name='conv2_2_dwise')
             .batch_normalization(relu=True, name='conv2_2_dwise_bn')
             .conv(1, 1, 24, 1, 1, biased=False, relu=False, name='conv2_2_linear')
             .batch_normalization(name='conv2_2_linear_bn')
             .conv(1, 1, 144, 1, 1, biased=False, relu=False, name='conv3_1_expand')
             .batch_normalization(relu=True, name='conv3_1_expand_bn')
             .conv(3, 3, 144, 1, 1, biased=False, group=144, relu=False, name='conv3_1_dwise')
             .batch_normalization(relu=True, name='conv3_1_dwise_bn')
             .conv(1, 1, 24, 1, 1, biased=False, relu=False, name='conv3_1_linear')
             .batch_normalization(name='conv3_1_linear_bn'))

            (self.feed('conv2_2_linear_bn',
                       'conv3_1_linear_bn')
             .add(name='block_3_1')
             .conv(1, 1, 144, 1, 1, biased=False, relu=False, name='conv3_2_expand')
             .batch_normalization(relu=True, name='conv3_2_expand_bn')
             .conv(3, 3, 144, 2, 2, biased=False, group=144, relu=False, name='conv3_2_dwise')
             .batch_normalization(relu=True, name='conv3_2_dwise_bn')
             .conv(1, 1, 32, 1, 1, biased=False, relu=False, name='conv3_2_linear')
             .batch_normalization(name='conv3_2_linear_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_1_expand')
             .batch_normalization(relu=True, name='conv4_1_expand_bn')
             .conv(3, 3, 192, 1, 1, biased=False, group=192, relu=False, name='conv4_1_dwise')
             .batch_normalization(relu=True, name='conv4_1_dwise_bn')
             .conv(1, 1, 32, 1, 1, biased=False, relu=False, name='conv4_1_linear')
             .batch_normalization(name='conv4_1_linear_bn'))

            (self.feed('conv3_2_linear_bn',
                       'conv4_1_linear_bn')
             .add(name='block_4_1')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_2_expand')
             .batch_normalization(relu=True, name='conv4_2_expand_bn')
             .conv(3, 3, 192, 1, 1, biased=False, group=192, relu=False, name='conv4_2_dwise')
             .batch_normalization(relu=True, name='conv4_2_dwise_bn')
             .conv(1, 1, 32, 1, 1, biased=False, relu=False, name='conv4_2_linear')
             .batch_normalization(name='conv4_2_linear_bn'))

            (self.feed('block_4_1',
                       'conv4_2_linear_bn')
             .add(name='block_4_2')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_3_expand')
             .batch_normalization(relu=True, name='conv4_3_expand_bn')
             .conv(3, 3, 192, 1, 1, biased=False, group=192, relu=False, name='conv4_3_dwise')
             .batch_normalization(relu=True, name='conv4_3_dwise_bn')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv4_3_linear')
             .batch_normalization(name='conv4_3_linear_bn')
             .conv(1, 1, 384, 1, 1, biased=False, relu=False, name='conv4_4_expand')
             .batch_normalization(relu=True, name='conv4_4_expand_bn')
             .conv(3, 3, 384, 1, 1, biased=False, group=384, relu=False, name='conv4_4_dwise')
             .batch_normalization(relu=True, name='conv4_4_dwise_bn')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv4_4_linear')
             .batch_normalization(name='conv4_4_linear_bn'))

            (self.feed('conv4_3_linear_bn',
                       'conv4_4_linear_bn')
             .add(name='block_4_4')
             .conv(1, 1, 384, 1, 1, biased=False, relu=False, name='conv4_5_expand')
             .batch_normalization(relu=True, name='conv4_5_expand_bn')
             .conv(3, 3, 384, 1, 1, biased=False, group=384, relu=False, name='conv4_5_dwise')
             .batch_normalization(relu=True, name='conv4_5_dwise_bn')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv4_5_linear')
             .batch_normalization(name='conv4_5_linear_bn'))

            (self.feed('block_4_4',
                       'conv4_5_linear_bn')
             .add(name='block_4_5')
             .conv(1, 1, 384, 1, 1, biased=False, relu=False, name='conv4_6_expand')
             .batch_normalization(relu=True, name='conv4_6_expand_bn')
             .conv(3, 3, 384, 1, 1, biased=False, group=384, relu=False, name='conv4_6_dwise')
             .batch_normalization(relu=True, name='conv4_6_dwise_bn')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv4_6_linear')
             .batch_normalization(name='conv4_6_linear_bn'))

            (self.feed('block_4_5',
                       'conv4_6_linear_bn')
             .add(name='block_4_6')
             .conv(1, 1, 384, 1, 1, biased=False, relu=False, name='conv4_7_expand')
             .batch_normalization(relu=True, name='conv4_7_expand_bn')
             .conv(3, 3, 384, 2, 2, biased=False, group=384, relu=False, name='conv4_7_dwise')
             .batch_normalization(relu=True, name='conv4_7_dwise_bn')
             .conv(1, 1, 96, 1, 1, biased=False, relu=False, name='conv4_7_linear')
             .batch_normalization(name='conv4_7_linear_bn')
             .conv(1, 1, 576, 1, 1, biased=False, relu=False, name='conv5_1_expand')
             .batch_normalization(relu=True, name='conv5_1_expand_bn')
             .conv(3, 3, 576, 1, 1, biased=False, group=576, relu=False, name='conv5_1_dwise')
             .batch_normalization(relu=True, name='conv5_1_dwise_bn')
             .conv(1, 1, 96, 1, 1, biased=False, relu=False, name='conv5_1_linear')
             .batch_normalization(name='conv5_1_linear_bn'))

            (self.feed('conv4_7_linear_bn',
                       'conv5_1_linear_bn')
             .add(name='feature'))

            feature_lv = 'feature'
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

