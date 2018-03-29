#!/bin/bash
arr="mobilenet_accurate mobilenet_v2 mobilenet_ms vgg16x4 vgg"
for model in $arr
do 
	python3 src/eval.py --model=$model
done

