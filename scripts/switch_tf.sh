
sudo pip3 uninstall tensorflow-gpu
#sudo pip3 install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp35-cp35m-linux_x86_64.whl
sudo pip3 install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0rc1-cp34-cp34m-linux_x86_64.whl 
python3 src/tensorrt_inference.py --graph=./models/graph/mobilenet_thin_432x368 --engine=reprodu.engine

