#!/bin/bash
#arr="mobilenet_accurate mobilenet_v2 mobilenet_ms vgg16x4 vgg"
#for model in $arr
#do 
#	python3 src/eval.py --model=$model
#done

python3 src/eval_caffemodel.py  --input-width=656 --input-height=368
python3 src/eval_caffemodel.py  --input-width=440 --input-height=256
python3 src/eval_caffemodel.py  --input-width=328 --input-height=184
