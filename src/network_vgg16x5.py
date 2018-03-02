import network_base
import tensorflow as tf

class VGG16x5Network(network_base.BaseNetwork):
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
                .conv(3, 3, 24, 1, 1, name='conv1_1')
                .conv(3, 3, 22, 1, 1, name='conv1_2')
                .max_pool(2, 2, 2, 2, name='pool1')
                .conv(3, 3, 41, 1, 1, name='conv2_1')
                .conv(3, 3, 51, 1, 1, name='conv2_2')
                .max_pool(2, 2, 2, 2, name='pool2')
                .conv(3, 3, 108, 1, 1, name='conv3_1')
                .conv(3, 3, 89, 1, 1, name='conv3_2')
                .conv(3, 3, 111, 1, 1, name='conv3_3')
                .max_pool(2, 2, 2, 2, name='pool3')
                .conv(3, 3, 184, 1, 1, name='conv4_1')
                .conv(3, 3, 276, 1, 1, name='conv4_2')
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
