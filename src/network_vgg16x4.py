
import network_base
import tensorflow as tf


class VGG16x4Network(network_base.BaseNetwork):
    def __init__(self, inputs, trainable=True, conv_width=1.0, conv_width2=0.75):
        self.conv_width = conv_width
        self.conv_width2 = conv_width2 if conv_width2 else conv_width
        network_base.BaseNetwork.__init__(self, inputs, trainable)

    def setup(self):
          min_depth = 8
          depth = lambda d: max(int(d * self.conv_width), min_depth)
          depth2 = lambda d: max(int(d * self.conv_width2), min_depth)
          (self.feed('image')
             .normalize_vgg(name='preprocess')
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 1, 22, 1, 1, relu=False, name='conv1_2_V')
             .conv(1, 3, 22, 1, 1, relu=False, name='conv1_2_H')
             .conv(1, 1, 59, 1, 1, name='conv1_2_P')
             .max_pool(2, 2, 2, 2, name='pool1_stage1')
             .conv(3, 1, 37, 1, 1, relu=False, name='conv2_1_V')
             .conv(1, 3, 37, 1, 1, relu=False, name='conv2_1_H')
             .conv(1, 1, 118, 1, 1, name='conv2_1_P')
             .conv(3, 1, 47, 1, 1, relu=False, name='conv2_2_V')
             .conv(1, 3, 47, 1, 1, relu=False, name='conv2_2_H')
             .conv(1, 1, 119, 1, 1, name='conv2_2_P')
             .max_pool(2, 2, 2, 2, name='pool2_stage1')
             .conv(3, 1, 83, 1, 1, relu=False, name='conv3_1_V')
             .conv(1, 3, 83, 1, 1, relu=False, name='conv3_1_H')
             .conv(1, 1, 226, 1, 1,  name='conv3_1_P')
             .conv(3, 1, 89, 1, 1, relu=False, name='conv3_2_V')
             .conv(1, 3, 89, 1, 1, relu=False, name='conv3_2_H')
             .conv(1, 1, 243, 1, 1, name='conv3_2_P')
             .conv(3, 1, 106, 1, 1, relu=False, name='conv3_3_V')
             .conv(1, 3, 106, 1, 1, relu=False, name='conv3_3_H')
             .conv(1, 1, 256, 1, 1, name='conv3_3_P')
             .max_pool(2, 2, 2, 2, name='pool3_stage1')
             .conv(3, 1, 175, 1, 1, relu=False, name='conv4_1_V')
             .conv(1, 3, 175, 1, 1, relu=False, name='conv4_1_H')
             .conv(1, 1, 482, 1, 1, name='conv4_1_P')
             # .conv(3, 1, 192, 1, 1, relu=False, name='conv4_2_V')
             # .conv(1, 3, 192, 1, 1, relu=False, name='conv4_2_H')
             # .conv(1, 1, 457, 1, 1, name='conv4_2_P')

             .conv(1, 3, depth2(256), 1, 1, name='conv4_3_CPM')
             .conv(1, 1, depth2(128), 1, 1, name='conv4_4_CPM')
             .conv(3, 3, depth2(128), 1, 1, name='conv5_1_CPM_L1')
             .conv(3, 3, depth2(128), 1, 1, name='conv5_2_CPM_L1')
             .conv(3, 3, depth2(128), 1, 1, name='conv5_3_CPM_L1')
             .conv(1, 1, depth2(512), 1, 1, name='conv5_4_CPM_L1')
             .conv(1, 1, 38, 1, 1, relu=False, name='conv5_5_CPM_L1'))

          (self.feed('conv4_4_CPM')
             .conv(3, 3, depth2(128), 1, 1, name='conv5_1_CPM_L2')
             .conv(3, 3, depth2(128), 1, 1, name='conv5_2_CPM_L2')
             .conv(3, 3, depth2(128), 1, 1, name='conv5_3_CPM_L2')
             .conv(1, 1, depth2(512), 1, 1, name='conv5_4_CPM_L2')
             .conv(1, 1, 19, 1, 1, relu=False, name='conv5_5_CPM_L2'))

          (self.feed('conv5_5_CPM_L1',
                   'conv5_5_CPM_L2',
                   'conv4_4_CPM')
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
                   'conv4_4_CPM')
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
                   'conv4_4_CPM')
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
                   'conv4_4_CPM')
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
                   'conv4_4_CPM')
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
        return None
