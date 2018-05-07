import tensorflow as tf
import time

import cv2
import numpy as np
import time
import logging
import argparse

from tensorflow.python.client import timeline

from common import  CocoPairsRender, read_imgfile, CocoColors
from estimator import PoseEstimator , TfPoseEstimator 
from networks import get_network
from pose_dataset import CocoPose

from tf_tensorrt_convert import * 

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True
#image_input = np.load('img.npy').transpose((2,0,1)).astype(np.float32).copy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Openpose Inference')
    parser.add_argument('--imgpath', type=str, default='./images/ski.jpg')
    parser.add_argument('--input-width', type=int, default=432)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--stage-level', type=int, default=6)
    parser.add_argument('--graph', type=str, default='./models/graph/mobilenet_thin_432x368/graph_opt.pb')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet / mobilenet_accurate / mobilenet_fast')
    parser.add_argument('--engine', type=str, default='mobilenet_thin.engine')
    parser.add_argument('--caffe', type=bool, default=False)
    parser.add_argument('--caffemodel', type=str, default='./models/pretrained/cmupose/pose_iter_440000.caffemodel')
    parser.add_argument('--proto', type=str, default='./models/pretrained/cmupose/pose_deploy_linevec.prototxt')
    args = parser.parse_args()

    with tf.Session(config=config) as sess:
        image = read_imgfile(args.imgpath, args.input_width, args.input_height)
        #image = (image /255.0 - 0.5 )*2
        image_input = image.transpose((2,0,1)).astype(np.float32).copy()
        print('image input dim is ', image_input.shape)
        if not args.caffe : 
            engine = create_engine(args.engine,  args.graph, args.input_height, args.input_width,  'image', 'Openpose/concat_stage7')
            output = tensorrt_inference(image_input, 57, args.input_height, args.input_width,engine)
            output = output.reshape(57, int(args.input_height/8), int(args.input_width/8)).transpose((1,2,0))
            heatMat, pafMat = output[:,:,:19], output[:,:,19:]

        else:
            engine = create_engine_from_caffe(args.engine, args.caffemodel, args.proto,  'image', 'net_output')
            output = tensorrt_inference(image_input / 256 - 0.5, 57, args.input_height, args.input_width, engine)
            output = output.reshape(57, int(args.input_height/8), int(args.input_width/8)).transpose((1,2,0))
            heatMat, pafMat = output[:,:,:19], output[:,:,19:]

        a = time.time()
        humans = PoseEstimator.estimate(heatMat, pafMat)
        logging.info('pose- elapsed_time={}'.format(time.time() - a))

        logging.info('image={} heatMap={} pafMat={}'.format(image.shape, heatMat.shape, pafMat.shape))
        process_img = CocoPose.display_image(image, heatMat, pafMat, as_numpy=True)
        np.save('heatmap_tf.npy', heatMat)
        # display
        image = cv2.imread(args.imgpath)
        image_h, image_w = image.shape[:2]
        image = TfPoseEstimator.draw_humans(image, humans)

        scale = 480.0 / image_h
        newh, neww = 480, int(scale * image_w + 0.5)

        image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)
        convas = np.zeros([480, 640 + neww, 3], dtype=np.uint8)
        convas[:, :640] = process_img
        convas[:, 640:] = image
        #cv2.imshow('result', convas)
        #cv2.waitKey(0)
        cv2.imwrite("tmp.jpg", convas)
        tf.train.write_graph(sess.graph_def, '.', 'graph-tmp.pb', as_text=True)
