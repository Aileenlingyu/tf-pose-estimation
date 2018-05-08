python3 src/inference.py --model=mobilenet_thin --input-width=800 --input-height=540
python3 -m tensorflow.python.tools.freeze_graph --input_graph graph-tmp.pb --output_graph graph_freeze.pb --input_checkpoint ./models/trained/mobilenet_thin_432x368/model-160001 --output_node_names=Openpose/concat_stage7
python3 src/tensorrt_inference.py --input-width=800 --input-height=540 --graph=graph_freeze.pb --engine=mobilenet_thin_800x540_half16.engine --half16=1

python3 src/inference.py --model=mobilenet_thin --input-width=656 --input-height=368
python3 -m tensorflow.python.tools.freeze_graph --input_graph graph-tmp.pb --output_graph graph_freeze.pb --input_checkpoint ./models/trained/mobilenet_thin_432x368/model-160001 --output_node_names=Openpose/concat_stage7
python3 src/tensorrt_inference.py --input-width=656 --input-height=$368 --graph=graph_freeze.pb --engine=mobilenet_thin_656x368_half16.engine --half16=1

python3 src/inference.py --model=mobilenet_thin --input-width=440 --input-height=256
python3 -m tensorflow.python.tools.freeze_graph --input_graph graph-tmp.pb --output_graph graph_freeze.pb --input_checkpoint ./models/trained/mobilenet_thin_432x368/model-160001 --output_node_names=Openpose/concat_stage7
python3 src/tensorrt_inference.py --input-width=440 --input-height=256 --graph=graph_freeze.pb --engine=mobilenet_thin_440x256_half16.engine --half16=1

python3 src/inference.py --model=mobilenet_thin --input-width=328 --input-height=184
python3 -m tensorflow.python.tools.freeze_graph --input_graph graph-tmp.pb --output_graph graph_freeze.pb --input_checkpoint ./models/trained/mobilenet_thin_432x368/model-160001 --output_node_names=Openpose/concat_stage7
python3 src/tensorrt_inference.py --input-width=328 --input-height=184 --graph=graph_freeze.pb --engine=mobilenet_thin_328x184_half16.engine --half16=1


python3 src/inference.py --model=mobilenet_original  --input-width=800 --input-height=540
python3 -m tensorflow.python.tools.freeze_graph --input_graph graph-tmp.pb --output_graph graph_freeze.pb --input_checkpoint ./models/trained/mobilenet_thin_benchmark/model-388003 --output_node_names=Openpose/concat_stage7
python3 src/tensorrt_inference.py --input-width=800 --input-height=540 --graph=./models/graph/mobilenet_thin_432x368/graph_opt.pb --engine=mobilenet_original_800x540_half16.engine  --half16=1

python3 src/inference.py --model=mobilenet_original --input-width=656 --input-height=368
python3 -m tensorflow.python.tools.freeze_graph --input_graph graph-tmp.pb --output_graph graph_freeze.pb --input_checkpoint ./models/trained/mobilenet_thin_benchmark/model-388003 --output_node_names=Openpose/concat_stage7
python3 src/tensorrt_inference.py --input-width=656 --input-height=368 --graph=graph_freeze.pb --engine=mobilenet_original_656x368_half16.engine  --half16=1

python3 src/inference.py --model=mobilenet_original --input-width=440 --input-height=256
python3 -m tensorflow.python.tools.freeze_graph --input_graph graph-tmp.pb --output_graph graph_freeze.pb --input_checkpoint ./models/trained/mobilenet_thin_benchmark/model-388003 --output_node_names=Openpose/concat_stage7
python3 src/tensorrt_inference.py --input-width=440 --input-height=256 --graph=graph_freeze.pb --engine=mobilenet_original_440x256_half16.engine  --half16=1

python3 src/inference.py --model=mobilenet_original --input-width=328 --input-height=184
python3 -m tensorflow.python.tools.freeze_graph --input_graph graph-tmp.pb --output_graph graph_freeze.pb --input_checkpoint ./models/trained/mobilenet_thin_benchmark/model-388003 --output_node_names=Openpose/concat_stage7
python3 src/tensorrt_inference.py --input-width=328 --input-height=184 --graph=graph_freeze.pb --engine=mobilenet_original_328x184_half16.engine  --half16=1
