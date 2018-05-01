import sys, os 
os.environ['GLOG_minloglevel'] = '2' 
import pickle
import cv2
import numpy as np
import time
import logging
import argparse
import caffe
from common import  CocoPairsRender, read_imgfile, CocoColors
from estimator import PoseEstimator , TfPoseEstimator 
from networks import get_network
from pose_dataset import CocoPose

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Openpose Inference')
    parser.add_argument('--imgpath', type=str, default='./images/p2.jpg')
    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--stage-level', type=int, default=6)
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet / mobilenet_accurate / mobilenet_fast')
    parser.add_argument('--caffemodel', type=str, default='./models/pretrained/cmupose/pose_iter_440000.caffemodel')
    parser.add_argument('--proto', type=str, default='./models/pretrained/cmupose/pose_deploy_linevec.prototxt')

    args = parser.parse_args()

    caffe.set_mode_gpu()
    net = caffe.Net(args.proto, args.caffemodel, caffe.TEST)
    image = cv2.imread(args.imgpath)
    image = cv2.resize(image, (args.input_width, args.input_height))

    net.blobs['image'].reshape(*(1, 3, image.shape[0], image.shape[1]))
    net.blobs['image'].data[...] = np.transpose(np.float32(image[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;

    out = net.forward()
    concat_stage7 = out['net_output'][0]
    heatMat, pafMat = concat_stage7[:19, :, :], concat_stage7[19:, :, :]

    humans = PoseEstimator.estimate(heatMat, pafMat)
    process_img = CocoPose.display_image(image, heatMat, pafMat, as_numpy=True)

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

    cv2.imwrite("cmupose.jpg", image)

