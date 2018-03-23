
IMAGE_FOLDER=/home/zaikun/hdd/data/keypoint/val2017
ANO_FOLDER=/home/zaikun/hdd/data/keypoint/annotations/person_keypoints_val2017.json
python3 src/eval.py --image_dir=$IMAGE_FOLDER --write_coco_json=$ANO_FOLDER --display 
