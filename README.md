# tf-pose-estimation

'Openpose' for human pose estimation have been implemented using Tensorflow. It also provides several variants that have made some changes to the network structure for **real-time processing on the CPU or low-power embedded devices.**

## Features in this fork
 - Add support for using pretrained model : mobilenet_V2, VGG16x4 (pruned VGG16 model, 4 times faster)
 - trainging with multi-scale loss
 - more mobilenet networks config
 - bug fix in the original repo
 - evalution script on validation dataset (support evaluation of both tf and caffe model)
 - tensorrt support 




##Inference From Caffemodel (cmu openpose)

python3 src/inference_cmupose.py  --input-width=656

## Inference From Mobilenet_thin 

python3 src/inference.py --input-width=656


## Inference From Tensorrt engine 

python3 src/tensorrt_inference.py --graph=yourgraph.opt.pb --engine=yourmodel.engine

## Inference and run from webcam With Tensorflow with tensorrt built-in

python3 src/run_webcam.py --use_tensorrt=1
