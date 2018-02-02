import argparse
import logging
import time

import cv2
import numpy as np
from pose_dataset import CocoPose

from estimator import TfPoseEstimator,PoseEstimator
from networks import get_graph_path, model_wh, get_network
import tensorflow as tf
logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--stage-level', type=int, default=6)
    parser.add_argument('--model', type=str, default='mobilenet_thin_432x368', help='cmu_640x480 / cmu_640x360 / mobilenet_thin_432x368')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    #logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    #w, h = model_wh(args.model)
    #e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    input_node = tf.placeholder(tf.float32, shape=(1, args.input_height, args.input_width, 3), name='image')
    with tf.Session(config=config) as sess:
        net, _, last_layer = get_network(args.model, input_node, sess)
        while True:
            ret_val, image = cam.read()

            logger.debug('image preprocess+')
            # if args.zoom < 1.0:
            #     canvas = np.zeros_like(image)
            #     img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            #     dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
            #     dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
            #     canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
            #     image = canvas
            # elif args.zoom > 1.0:
            #     img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            #     dx = (img_scaled.shape[1] - image.shape[1]) // 2
            #     dy = (img_scaled.shape[0] - image.shape[0]) // 2
            #     image = img_scaled[dy:image.shape[0], dx:image.shape[1]]

            resized_image = cv2.resize(image, (args.input_height, args.input_width), interpolation=cv2.INTER_AREA)

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            pafMat, heatMat = sess.run(
                [
                    net.get_output(name=last_layer.format(stage=args.stage_level, aux=1)),
                    net.get_output(name=last_layer.format(stage=args.stage_level, aux=2))
                ], feed_dict={'image:0': [resized_image]}, options=run_options, run_metadata=run_metadata
            )
            heatMat, pafMat = heatMat[0], pafMat[0]

            humans = PoseEstimator.estimate(heatMat, pafMat)

            logging.info('image={} heatMap={} pafMat={}'.format(resized_image.shape, heatMat.shape, pafMat.shape))
            process_img = CocoPose.display_image(resized_image, heatMat, pafMat, as_numpy=True)

            # display
            image_h, image_w = image.shape[:2]
            image = TfPoseEstimator.draw_humans(image, humans)
            #
            scale = 480.0 / image_h
            newh, neww = 480, int(scale * image_w + 0.5)

            image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)

            convas = np.zeros([480, 640 + neww, 3], dtype=np.uint8)
            convas[:, :640] = process_img
            convas[:, 640:] = image

            cv2.imshow('result', convas)
            cv2.waitKey(1)


    cv2.destroyAllWindows()
