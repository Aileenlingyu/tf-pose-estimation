
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --model=mobilenet_zaikun_side --datapath=/home/zaikun/hdd/data/keypoint/annotations/  --lr=0.001 --imgpath=/home/zaikun/hdd/data/keypoint/ --logpath=./logs/ --gpus=4 --modelpath=./model/ --batchsize=32  --input-height=368 --input-width=368
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --model=vgg16x4_stage2 --datapath=/home/zaikun/hdd/data/keypoint/annotations/  --lr=0.0001 --imgpath=/home/zaikun/hdd/data/keypoint/ --logpath=./logs/ --gpus=4 --modelpath=./model/ --batchsize=32  --input-height=368 --input-width=368
