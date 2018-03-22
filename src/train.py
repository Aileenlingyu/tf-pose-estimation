import matplotlib as mpl
mpl.use('Agg')      # training mode, no screen should be open. (It will block training loop)

import argparse
import logging
import os
import time


import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorpack.dataflow.remote import RemoteDataZMQ

from pose_dataset import get_dataflow_batch, DataFlowToQueue, CocoPose
from pose_augment import set_network_input_wh, set_network_scale, set_ms
from common import get_sample_images
from networks import get_network

logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--model', default='mobilenet', help='model name')
    parser.add_argument('--datapath', type=str, default='/root/coco/annotations')
    parser.add_argument('--imgpath', type=str, default='/root/coco/')
    parser.add_argument('--batchsize', type=int, default=96)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max-epoch', type=int, default=90)
    parser.add_argument('--lr', type=str, default='0.01')
    parser.add_argument('--modelpath', type=str, default='/data/private/tf-openpose-models-2018-1/')
    parser.add_argument('--logpath', type=str, default='/data/private/tf-openpose-log-2018-1/')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--remote-data', type=str, default='', help='eg. tcp://0.0.0.0:1027')
    parser.add_argument('--decay_steps', type=int, default = 10000)
    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--parts', type=int, default=19)
    parser.add_argument('--do_ms', type=bool, default=False)
    parser.add_argument('--lr_constant', type=bool, default=True)

    args = parser.parse_args()

    if args.gpus <= 0:
        raise Exception('gpus <= 0')

    # define input placeholder
    set_network_input_wh(args.input_width, args.input_height)
    scale = 4
    ms = args.do_ms
    if args.model in ['cmu', 'vgg', 'mobilenet_thin_dilate' ,'mobilenet_thin', 'mobilenet_thin_up' , 'mobilenet_thin_shortcut',
                      'vgg16x4', 'vgg16x4_stage2', 'mobilenet_fast', 'mobilenet_ms' , 'mobilenet_accurate', 'resnet32',
                      'mobilenet_v2', 'mobilenet_thin_fatbranch', 'mobilenet_zaikun', 'mobilenet_zaikun_side']:
        scale = 8

    set_network_scale(scale)
    set_ms(ms)

    output_w, output_h = args.input_width // scale, args.input_height // scale

    logger.info('define model+')
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
        input_node = tf.placeholder(tf.float32, shape=(args.batchsize, args.input_height, args.input_width, 3), name='image')
        vectmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, args.parts * 2), name='vectmap')
        heatmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, args.parts), name='heatmap')
        two_vectormap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h * 2, output_w * 2, args.parts * 2), name = "two_vectmap")
        two_headmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h * 2, output_w * 2, args.parts), name = "two_heatmap")

        # prepare data
        if not args.remote_data:
            df = get_dataflow_batch(args.datapath, True, args.batchsize, img_path=args.imgpath)
        else:
            # transfer inputs from ZMQ
            df = RemoteDataZMQ(args.remote_data, hwm=3)

        if ms :
            enqueuer = DataFlowToQueue(df, [input_node, heatmap_node, vectmap_node, two_headmap_node, two_vectormap_node], queue_size=100)
            q_inp, q_heat, q_vect, q_two_heat, q_two_vect = enqueuer.dequeue()
        else:
            enqueuer = DataFlowToQueue(df, [input_node, heatmap_node, vectmap_node], queue_size=100)
            q_inp, q_heat, q_vect = enqueuer.dequeue()
    df_valid = get_dataflow_batch(args.datapath, False, args.batchsize, img_path=args.imgpath)
    df_valid.reset_state()
    validation_cache = []

    val_image = get_sample_images(args.input_width, args.input_height)
    logger.info('tensorboard val image: %d' % len(val_image))
    logger.info('scale is %d' % scale)
    logger.info(q_inp)
    logger.info(q_heat)
    logger.info(q_vect)
    logger.info(ms)
    # define model for multi-gpu
    if not ms :
        q_inp_split, q_heat_split, q_vect_split = tf.split(q_inp, args.gpus), tf.split(q_heat, args.gpus), tf.split(q_vect, args.gpus)
    else:
        q_inp_split, q_heat_split, q_vect_split, \
        q_two_heat_split, q_two_vect_slit = tf.split(q_inp, args.gpus)\
                                          , tf.split(q_heat, args.gpus)\
                                          , tf.split(q_vect, args.gpus), tf.split(q_two_heat, args.gpus), tf.split(q_two_vect, args.gpus)

    output_vectmap = []
    output_heatmap = []
    losses = []
    last_losses_l1 = []
    last_losses_l2 = []

    output_two_heatmap = []
    output_two_vectmap = []
    last_losses_two_l1 = []
    last_losses_two_l2 = []

    outputs = []
    for gpu_id in range(args.gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                net, pretrain_path, last_layer = get_network(args.model, q_inp_split[gpu_id])

                if not ms:
                    vect, heat = net.loss_last()
                    output_vectmap.append(vect)
                    output_heatmap.append(heat)
                    outputs.append(net.get_output())

                    l1s, l2s = net.loss_l1_l2()
                    for idx, (l1, l2) in enumerate(zip(l1s, l2s)):
                        loss_l1 = tf.nn.l2_loss(tf.concat(l1, axis=0) - q_vect_split[gpu_id], name='loss_l1_stage%d_tower%d' % (idx, gpu_id))
                        loss_l2 = tf.nn.l2_loss(tf.concat(l2, axis=0) - q_heat_split[gpu_id], name='loss_l2_stage%d_tower%d' % (idx, gpu_id))
                        losses.append(tf.reduce_mean([loss_l1, loss_l2]))

                    last_losses_l1.append(loss_l1)
                    last_losses_l2.append(loss_l2)
                else:
                    vect, heat, two_vect, two_heat = net.loss_last()
                    output_vectmap.append(vect)
                    output_heatmap.append(heat)
                    output_two_heatmap.append(two_heat)
                    output_two_vectmap.append(two_vect)
                    outputs.append(net.get_output())
                    l1s, l2s, two_l1s, two_l2s = net.loss_l1_l2()
                    for idx, (l1, l2, two_l1, two_l2) in enumerate(zip(l1s, l2s, two_l1s, two_l2s)):
                        #import pdb; pdb.set_trace();
                        loss_l1 = tf.nn.l2_loss(tf.concat(l1, axis=0) - q_vect_split[gpu_id],
                                                name='loss_l1_stage%d_tower%d' % (idx, gpu_id))
                        loss_l2 = tf.nn.l2_loss(tf.concat(l2, axis=0) - q_heat_split[gpu_id],
                                                name='loss_l2_stage%d_tower%d' % (idx, gpu_id))
                        loss_two_l1 = tf.nn.l2_loss(tf.concat(two_l1, axis=0)-q_two_vect_slit[gpu_id],
                                                name='loss_two_l1_stage%d_tower%d' %(idx, gpu_id))
                        loss_two_l2 = tf.nn.l2_loss(tf.concat(two_l2, axis=0) - q_two_heat_split[gpu_id],
                                                name='loss_two_l2_stage%d_tower%d' % (idx, gpu_id))
                        losses.append(tf.reduce_mean([loss_l1, loss_l2, loss_two_l1, loss_two_l2]))

                    last_losses_l1.append(loss_l1)
                    last_losses_l2.append(loss_l2)
                    last_losses_two_l1.append(loss_two_l1)
                    last_losses_two_l2.append(loss_two_l2)

    outputs = tf.concat(outputs, axis=0)

    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
        # define loss
        if not ms:
            total_loss = tf.reduce_sum(losses) / args.batchsize
            total_loss_ll_paf = tf.reduce_sum(last_losses_l1) / args.batchsize
            total_loss_ll_heat = tf.reduce_sum(last_losses_l2) / args.batchsize
            total_loss_ll  = tf.reduce_sum([total_loss_ll_heat, total_loss_ll_paf])
        else:
            total_loss = tf.reduce_sum(losses) / args.batchsize
            total_loss_ll_paf = tf.reduce_sum(last_losses_l1) / args.batchsize
            total_loss_ll_heat = tf.reduce_sum(last_losses_l2) / args.batchsize
            total_loss_two_paf = tf.reduce_sum(last_losses_two_l1) / args.batchsize
            total_loss_two_heat =tf.reduce_sum(last_losses_two_l2) / args.batchsize
            total_loss_ll  = tf.reduce_sum([total_loss_ll_heat, total_loss_ll_paf, total_loss_two_paf, total_loss_two_heat])

        # define optimizer
        step_per_epoch = 121745 // args.batchsize
        global_step = tf.Variable(0, trainable=False)
        if ',' not in args.lr:
            starter_learning_rate = float(args.lr)
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       decay_steps=args.decay_steps * 96/ args.batchsize, decay_rate=1.0, staircase=True)
        else:
            lrs = [float(x) for x in args.lr.split(',')]
            boundaries = [step_per_epoch * 5 * i for i, _ in range(len(lrs)) if i > 0]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, lrs)

    #optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.0005, momentum=0.9, epsilon=1e-10)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss, global_step, colocate_gradients_with_ops=True)
    logger.info('define model-')

    # define summary
    tf.summary.scalar("loss", total_loss)
    tf.summary.scalar("loss_lastlayer", total_loss_ll)
    tf.summary.scalar("loss_lastlayer_paf", total_loss_ll_paf)
    tf.summary.scalar("loss_lastlayer_heat", total_loss_ll_heat)
    if ms:
        tf.summary.scalar("loss_lastlayer_two_paf", total_loss_two_paf)
        tf.summary.scalar("loss_lastlayer_two_heat", total_loss_two_heat)

    tf.summary.scalar("queue_size", enqueuer.size())
    merged_summary_op = tf.summary.merge_all()

    valid_loss = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll_paf = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll_heat = tf.placeholder(tf.float32, shape=[])
    if ms:
        valid_loss_two_heat = tf.placeholder(tf.float32, shape=[])
        valid_loss_two_paf = tf.placeholder(tf.float32, shape=[])

    sample_train = tf.placeholder(tf.float32, shape=(4, 640, 640, 3))
    sample_valid = tf.placeholder(tf.float32, shape=(12, 640, 640, 3))
    train_img = tf.summary.image('training sample', sample_train, 4)
    valid_img = tf.summary.image('validation sample', sample_valid, 12)
    valid_loss_t = tf.summary.scalar("loss_valid", valid_loss)
    valid_loss_ll_t = tf.summary.scalar("loss_valid_lastlayer", valid_loss_ll)
    merged_validate_op = tf.summary.merge([train_img, valid_img, valid_loss_t, valid_loss_ll_t])

    saver = tf.train.Saver(max_to_keep=100)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        training_name = '{}_batch:{}_lr:{}_gpus:{}_{}x{}_{}'.format(
            args.model,
            args.batchsize,
            args.lr,
            args.gpus,
            args.input_width, args.input_height,
            args.tag
        )
        logger.info('model weights initialization')
        sess.run(tf.global_variables_initializer())

        if args.checkpoint:
            logger.info('Restore from checkpoint...')
            # loader = tf.train.Saver(net.restorable_variables())
            # loader.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
            saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
            logger.info('Restore from checkpoint...Done')
        elif pretrain_path:
            logger.info('Restore pretrained weights...')
            if '.ckpt' in pretrain_path:
                loader = tf.train.Saver(net.restorable_variables())
                loader.restore(sess, pretrain_path)
            elif '.npy' in pretrain_path:
                net.load(pretrain_path, sess, True)
            else:
                logger.info('training from scratch');
        logger.info('prepare file writer')
        file_writer = tf.summary.FileWriter(args.logpath + training_name, sess.graph)

        logger.info('prepare coordinator')
        coord = tf.train.Coordinator()
        enqueuer.set_coordinator(coord)
        enqueuer.start()

        logger.info('Training Started.')
        time_started = time.time()
        last_gs_num = last_gs_num2 = 0
        initial_gs_num = sess.run(global_step)

        while True:
            _, gs_num = sess.run([train_op, global_step])

            if gs_num > step_per_epoch * args.max_epoch:
                break

            if gs_num - last_gs_num >= 100:
                if not ms:
                    train_loss, train_loss_ll, train_loss_ll_paf, train_loss_ll_heat, lr_val, summary, queue_size = sess.run([total_loss, total_loss_ll, total_loss_ll_paf, total_loss_ll_heat, learning_rate, merged_summary_op, enqueuer.size()])
                else:
                    train_loss, train_loss_ll, train_loss_ll_paf, train_loss_ll_heat, train_loss_two_paf, train_loss_two_heat, lr_val, summary, queue_size = sess.run([total_loss, total_loss_ll, total_loss_ll_paf, total_loss_ll_heat, total_loss_two_paf, total_loss_two_heat, learning_rate, merged_summary_op, enqueuer.size()])

                # log of training loss / accuracy
                batch_per_sec = (gs_num - initial_gs_num) / (time.time() - time_started)
                if not ms:
                    logger.info('epoch=%.2f step=%d, %0.4f examples/sec lr=%f, loss=%g, loss_ll=%g, loss_ll_paf=%g, loss_ll_heat=%g, q=%d' % (gs_num / step_per_epoch, gs_num, batch_per_sec * args.batchsize, lr_val, train_loss, train_loss_ll, train_loss_ll_paf, train_loss_ll_heat, queue_size))
                else:
                    logger.info('epoch=%.2f step=%d, %0.4f examples/sec lr=%f, loss=%g, loss_ll=%g, loss_ll_paf=%g, loss_ll_heat=%g, loss_ll_twopaf=%g, loss_ll_twoheat=%g,  q=%d' % (gs_num / step_per_epoch, gs_num, batch_per_sec * args.batchsize, lr_val, train_loss, train_loss_ll, train_loss_ll_paf, train_loss_ll_heat, train_loss_two_paf, train_loss_two_heat, queue_size))
                last_gs_num = gs_num

                file_writer.add_summary(summary, gs_num)

            if gs_num - last_gs_num2 >= 1000:
                # save weights
                saver.save(sess, os.path.join(args.modelpath, training_name, 'model'), global_step=global_step)

                average_loss = average_loss_ll = average_loss_ll_paf = average_loss_ll_heat = 0
                total_cnt = 0

                if len(validation_cache) == 0:
                    if not ms :
                        for images_test, heatmaps, vectmaps in tqdm(df_valid.get_data()):
                            validation_cache.append((images_test, heatmaps, vectmaps))
                    else:
                        for image_test, heatmaps, vectmaps, two_heatmaps, two_vectmaps in tqdm(df_valid.get_data()):
                            validation_cache.append((image_test, heatmaps, vectmaps))

                    df_valid.reset_state()
                    del df_valid
                    df_valid = None

                # log of test accuracy
                for images_test, heatmaps, vectmaps in validation_cache:
                    lss, lss_ll, lss_ll_paf, lss_ll_heat, vectmap_sample, heatmap_sample = sess.run(
                        [total_loss, total_loss_ll, total_loss_ll_paf, total_loss_ll_heat, output_vectmap, output_heatmap],
                        feed_dict={q_inp: images_test, q_vect: vectmaps, q_heat: heatmaps}
                    )
                    average_loss += lss * len(images_test)
                    average_loss_ll += lss_ll * len(images_test)
                    average_loss_ll_paf += lss_ll_paf * len(images_test)
                    average_loss_ll_heat += lss_ll_heat * len(images_test)
                    total_cnt += len(images_test)

                logger.info('validation(%d) %s loss=%f, loss_ll=%f, loss_ll_paf=%f, loss_ll_heat=%f' % (total_cnt, training_name, average_loss / total_cnt, average_loss_ll / total_cnt, average_loss_ll_paf / total_cnt, average_loss_ll_heat / total_cnt))


                last_gs_num2 = gs_num

                sample_image = [enqueuer.last_dp[0][i] for i in range(4)]
                outputMat = sess.run(
                    outputs,
                    feed_dict={q_inp: np.array((sample_image + val_image)*(args.batchsize // 16))}
                )
                pafMat, heatMat = outputMat[:, :, :, args.parts:], outputMat[:, :, :, :args.parts]

                sample_results = []
                for i in range(len(sample_image)):
                    test_result = CocoPose.display_image(sample_image[i], heatMat[i], pafMat[i], as_numpy=True)
                    test_result = cv2.resize(test_result, (640, 640))
                    test_result = test_result.reshape([640, 640, 3]).astype(float)
                    sample_results.append(test_result)

                test_results = []
                for i in range(len(val_image)):
                    test_result = CocoPose.display_image(val_image[i], heatMat[len(sample_image) + i], pafMat[len(sample_image) + i], as_numpy=True)
                    test_result = cv2.resize(test_result, (640, 640))
                    test_result = test_result.reshape([640, 640, 3]).astype(float)
                    test_results.append(test_result)

                # save summary
                summary = sess.run(merged_validate_op, feed_dict={
                    valid_loss: average_loss / total_cnt,
                    valid_loss_ll: average_loss_ll / total_cnt,
                    valid_loss_ll_paf: average_loss_ll_paf / total_cnt,
                    valid_loss_ll_heat: average_loss_ll_heat / total_cnt,
                    sample_valid: test_results,
                    sample_train: sample_results
                })
                file_writer.add_summary(summary, gs_num)

        saver.save(sess, os.path.join(args.modelpath, training_name, 'model'), global_step=global_step)
    logger.info('optimization finished. %f' % (time.time() - time_started))
