from common import  CocoPairsRender, read_imgfile, CocoColors
from estimator import PoseEstimator , TfPoseEstimator 
from pose_dataset import CocoPose
import glob
from random import shuffle
import numpy as np
from PIL import Image
import time, os
import logging
import argparse
import tensorrt as trt
import calibrator    #calibrator.py
import cv2 

DATA_DIR = '/home/zaikun/hdd/data/coco_gender/val_2017/'
CALIBRATION_DATASET_LOC = DATA_DIR + '*.jpg'

def create_calibration_dataset():
    calibration_files = glob.glob(CALIBRATION_DATASET_LOC)
    shuffle(calibration_files)
    return calibration_files[:100]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorrt int8 Inference')
    parser.add_argument('--imgpath', type=str, default='./images/p2.jpg')
    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--engine', type=str, default='mobilepose_int8.engine')
    args = parser.parse_args()
    calibration_files = create_calibration_dataset()
    batchstream = calibrator.ImageBatchStream(1, calibration_files, args.input_height, args.input_width)
    int8_calibrator = calibrator.PythonEntropyCalibrator(["image"], batchstream)
    engine = trt.lite.Engine(framework="c1",
                           deployfile="./models/pretrained/cmupose/pose_deploy_linevec.prototxt",
                           modelfile= "./models/pretrained/cmupose/pose_iter_440000.caffemodel",
                           max_batch_size=10,
                           max_workspace_size=(256 << 20),
                           input_nodes={"image":(3, args.input_height, args.input_width)},
                           output_nodes=["net_output"],
                           #preprocessors={"image":sub_mean_chw},
                           # postprocessors={"score":color_map},
                           data_type=trt.infer.DataType.INT8,
                           calibrator=int8_calibrator,
                           logger_severity=trt.infer.LogSeverity.INFO)
    image = cv2.imread(args.imgpath)
    image = cv2.imresize(image, (args.input_width, args.input_height))
    concat_stage7 = engine.infer(image)[0]
    heatMat, pafMat = concat_stage7[:19, :, :], concat_stage7[19:, :, :]
    humans = PoseEstimator.estimate(heatMat, pafMat)
    process_img = CocoPose.display_image(image, heatMat, pafMat, as_numpy=True)
    image = cv2.imread(args.imgpath)
    image_h, image_w = image.shape[:2]
    image = TfPoseEstimator.draw_humans(image, humans)
    scale = 480.0 / image_h
    newh, neww = 480, int(scale * image_w + 0.5)
    image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)
    convas = np.zeros([480, 640 + neww, 3], dtype=np.uint8)
    convas[:, :640] = process_img
    convas[:, 640:] = image
    cv2.imwrite("cmupose_int8.jpg", convas)
