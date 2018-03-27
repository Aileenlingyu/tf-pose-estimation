
import pickle
import tensorflow as tf
import cv2
import numpy as np
import time
import logging
import argparse
import json
from tensorflow.python.client import timeline

from common import  CocoPairsRender, read_imgfile, CocoColors
from estimator import PoseEstimator , TfPoseEstimator
from networks import get_network
from pose_dataset import CocoPose

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True


'''
    openpose impl of save to coco 
    
        const auto numberPeople = poseKeypoints.getSize(0);
            const auto numberBodyParts = poseKeypoints.getSize(1);
            const auto imageId = getLastNumber(imageName);
            for (auto person = 0 ; person < numberPeople ; person++)
            {
                // Comma at any moment but first element
                if (mFirstElementAdded)
                {
                    mJsonOfstream.comma();
                    mJsonOfstream.enter();
                }
                else
                    mFirstElementAdded = true;

                // New element
                mJsonOfstream.objectOpen();

                // image_id
                mJsonOfstream.key("image_id");
                mJsonOfstream.plainText(imageId);
                mJsonOfstream.comma();

                // category_id
                mJsonOfstream.key("category_id");
                mJsonOfstream.plainText("1");
                mJsonOfstream.comma();

                // keypoints - i.e. poseKeypoints
                mJsonOfstream.key("keypoints");
                mJsonOfstream.arrayOpen();
                const std::vector<int> indexesInCocoOrder{0, 15, 14, 17, 16,        5, 2, 6, 3, 7,        4, 11, 8, 12, 9,        13, 10};
                for (auto bodyPart = 0u ; bodyPart < indexesInCocoOrder.size() ; bodyPart++)
                {
                    const auto finalIndex = 3*(person*numberBodyParts + indexesInCocoOrder.at(bodyPart));
                    mJsonOfstream.plainText(poseKeypoints[finalIndex]);
                    mJsonOfstream.comma();
                    mJsonOfstream.plainText(poseKeypoints[finalIndex+1]);
                    mJsonOfstream.comma();
                    mJsonOfstream.plainText(1);
                    if (bodyPart < indexesInCocoOrder.size() - 1u)
                        mJsonOfstream.comma();
                }
                mJsonOfstream.arrayClose();


'''

def write_coco_json(humans, image_w, image_h):
    keypoints = []
    if len(humans) == 0:
        return [0] * 51
    for human in humans:
        for i in [0, 15,14,17,16,5, 2, 6, 3,7,4, 11, 8, 12, 9,13,10]:
            if i not in human.body_parts.keys():
                keypoints.extend([0,0, 0])
                continue
            body_part = human.body_parts[i]
            center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
            keypoints.append(center[0])
            keypoints.append(center[1])
            keypoints.append(1)
    return keypoints

def getLastName(file_name):
    return float(file_name.split('.')[0].strip("0"))

def load_coco(json_file):
    data =json.load(open(json_file))
    num_images = len(data['images'])
    result = {
        'image_id' : [],
        'name':[],
    }
    for i in range(num_images):
        #keypoints = data['annotations'][i]['keypoints']
        image_name = data['images'][i]['file_name']
        result['image_id'].append(getLastName(image_name))
        result['name'].append(image_name)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Openpose Inference')
    parser.add_argument('--imgpath', type=str, default='./images/p2.jpg')
    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--stage-level', type=int, default=6)
    parser.add_argument('--model', type=str, default='mobilenet', help='cmu / mobilenet / mobilenet_accurate / mobilenet_fast')
    parser.add_argument('--image_dir', type=str, default='/home/zaikun/hdd/data/keypoint/val2017/')
    parser.add_argument('--coco_json_file', type = str, default = '/home/zaikun/hdd/data/keypoint/annotations/person_keypoints_val2017.json')
    parser.add_argument('--write_coco_json', type=str, default='eval.json', help='the output json file')
    parser.add_argument('--display', type=bool, default=False)
    args = parser.parse_args()

    input_node = tf.placeholder(tf.float32, shape=(1, args.input_height, args.input_width, 3), name='image')
    fp = open(args.write_coco_json, 'w')
    result = []
    with tf.Session(config=config) as sess:
        net, _, last_layer = get_network(args.model, input_node, sess)
        coco_label = load_coco(args.coco_json_file)
        for i, img in enumerate(coco_label['name']):
            item = {
                'image_id' : 1,
                'category_id' : 1,
                'keypoints' :[],
                'score' : 0.0
            }
            image_id = coco_label['image_id'][i]

            img_name = args.image_dir + img
            image = read_imgfile(img_name, args.input_width, args.input_height)
            vec = sess.run(net.get_output(name='concat_stage7'), feed_dict={'image:0': [image]})

            a = time.time()
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            pafMat, heatMat = sess.run(
                [
                    net.get_output(name=last_layer.format(stage=args.stage_level, aux=1)),
                    net.get_output(name=last_layer.format(stage=args.stage_level, aux=2))
                ], feed_dict={'image:0': [image]}, options=run_options, run_metadata=run_metadata
            )

            heatMat, pafMat = heatMat[0], pafMat[0]

            '''
                Q:
                    Given n persons in one image, how do we output the result to a json ?
                    
                A:
                    In the coco validation images, images with multiple persons are written to
                    mutiple json items and in evaluation, the cocoapi will do the search for the 
                    right match for us.
                        
            '''
            a = time.time()
            humans = PoseEstimator.estimate(heatMat, pafMat)
            r = write_coco_json(humans, args.input_height, args.input_width)
            item['keypoints'] = r
            item['image_id']  = int(image_id)
            result.append(item)

            if i == 100:
                break

        json.dump(result,fp)


