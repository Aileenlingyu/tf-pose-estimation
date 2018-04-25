#usage : sh convert_pb_opt.sh input_cpkt  input-pb freeze-pb opt-pb

python3 /usr/local/lib/python3.5/dist-packages/tensorflow/python/tools/freeze_graph.py --input_graph $2 --output_graph $3 --input_checkpoint $1 --output_node_names=Openpose/concat_stage7

python3 /usr/local/lib/python3.5/dist-packages/tensorflow/python/tools/optimize_for_inference.py --input $3  --output $4 --input_names=image --output_names=Openpose/concat_stage7

