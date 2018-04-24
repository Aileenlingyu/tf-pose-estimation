import numpy as np
import json
import cv2
from pycocotools.coco import COCO
import os , sys
train_json = '/home/zaikun/hdd/data/keypoint/annotations/person_keypoints_train2017.json'
val_json = '/home/zaikun/hdd/data/keypoint/annotations/person_keypoints_val2017.json'
val_dir = '/home/zaikun/hdd/data/keypoint/val2017/'
train_dir = '/home/zaikun/hdd/data/keypoint/train2017/'

val_save_dir = '/home/zaikun/hdd/data/coco_gender/val_2017'
train_save_dir = '/home/zaikun/hdd/data/coco_gender/train_2017'

f = val_json
img_dir = val_dir
save_dir = val_save_dir


def save_coco_gender(f, save_dir, img_dir):
    print('loading   json file...')
    coco = COCO(f)
    keys = list(coco.imgs.keys())
    print('#of keys {}'.format(len(keys)))


    for i, ind in enumerate(keys):
        img_meta = coco.imgs[ind]
        img_idx = img_meta['id']
        img_name = img_meta['file_name']
        ann_idx = coco.getAnnIds(imgIds=img_idx)
        anns = coco.loadAnns(ann_idx)
        img = cv2.imread(img_dir + img_name)
        for ann in anns :
            if ann['category_id'] == 1 :
                id = ann['id']
                uid = img_name.split('.')[0] + '_' + str(id)
                bbox = ann['bbox']
                if bbox[2] < 30 or bbox[3] < 30 :
                    continue
                crop_img = img[int(bbox[1]): int(bbox[3] + bbox[1]), int(bbox[0]): int(bbox[0] + bbox[2])]
                output_name = os.path.join(save_dir, uid + '.jpg')
                import pdb; pdb.set_trace()
                cv2.imwrite(output_name, crop_img)
        if i % 1000 == 0:
            print(i)
                #cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2] + bbox[0]), int(bbox[3] + bbox[1])), (255, 0, 0), 5)


if __name__ == '__main__':
    save_coco_gender(val_json, val_save_dir, val_dir)
    #save_coco_gender(train_json, train_save_dir, train_dir)
