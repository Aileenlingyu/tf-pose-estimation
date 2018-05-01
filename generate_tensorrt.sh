
python3 src/inference.py --model=mobilenet_thin --input-width=$1 --input-height=$2

python3 -m tensorflow.python.tools.freeze_graph --input_graph graph-tmp.pb --output_graph graph_freeze.pb --input_checkpoint ./models/trained/mobilenet_thin_432x368/model-160001 --output_node_names=Openpose/concat_stage7

$HOME/hdd/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph     --in_graph=graph_freeze.pb    --out_graph=graph_opt.pb    --inputs='image'     --outputs='Openpose/concat_stage7'     --transforms='
    strip_unused_nodes(type=float, shape="1,$2,$1,3")
    remove_nodes(op=Identity, op=CheckNumerics)
    fold_constants(ignoreError=False)
    fold_old_batch_norms
    fold_batch_norms'

python3 src/tensorrt_inference.py --input-width=$1 --input-height=$2 --graph=graph_opt.pb --engine=mobilenet_thin_$1x$2.engine
