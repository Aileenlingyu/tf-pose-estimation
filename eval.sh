#!/bin/bash
arr="mobilenet_original mobilenet_thin"
for model in $arr
do 
	python3 src/eval.py --model=$model  --input-width=656 --input-height=368
	python3 src/eval.py --model=$model  --input-width=440 --input-height=256
	python3 src/eval.py --model=$model  --input-width=328 --input-height=184 
done

#CUDA_VISIBLE_DEVICES=0 python3 src/eval_caffemodel.py  --input-width=656 --input-height=368
#CUDA_VISIBLE_DEVICES=0 python3 src/eval_caffemodel.py  --input-width=440 --input-height=256
#CUDA_VISIBLE_DEVICES=0 python3 src/eval_caffemodel.py  --input-width=328 --input-height=184
