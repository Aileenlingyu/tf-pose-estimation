import tensorrt as trt
from tensorrt.lite import Engine
import time
import uff, os
from PIL import Image 
import pycuda.driver as cuda
import pycuda.autoinit
from tensorrt.parsers import uffparser
import numpy as np

#img = np.load('img.npy').transpose((2, 0, 1)).astype(np.float32).copy()
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

def analyze(data):
	return data

def create_engine(img,name, uff_name, model_path, c_o, height, width,  output_npy = 'conv0.npy', layer_name = 'MobilenetV1/Conv2d_0/Relu'):
	if not os.path.exists(name):
		# Load your newly created Tensorflow frozen model and convert it to UFF
		#import pdb; pdb.set_trace();
		uff_model = uff.from_tensorflow_frozen_model(model_path, [ layer_name ])#, output_filename = 'mobilepose.uff')
		dump = open(uff_name, 'wb')
		dump.write(uff_model)
		dump.close()
		# Create a UFF parser to parse the UFF file created from your TF Frozen model
		parser = uffparser.create_uff_parser()
		parser.register_input("image", (3, height, width),0)
		parser.register_output(layer_name)

		# Build your TensorRT inference engine
		# This step performs (1) Tensor fusion (2) Reduced precision 
		# (3) Target autotuning (4) Tensor memory management
		engine = trt.utils.uff_to_trt_engine(G_LOGGER, 
						     uff_model,
						     parser,
						     1,
						     1<<20) 
		trt.utils.write_engine_to_file(name, engine.serialize())
	else : 
		#engine = Engine(PLAN=name, postproceessors = {'Openpose/concat_stage7': analyze})
		engine = trt.utils.load_engine(G_LOGGER, name)

	print("engine created ....")	
	runtime = trt.infer.create_infer_runtime(G_LOGGER)
	context = engine.create_execution_context()
	
	output = np.empty(c_o * height* width , dtype = np.float32)
	
	#alocate device memory
	d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
	d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
	bindings = [int(d_input), int(d_output)]
	stream = cuda.Stream()
	#transfer input data to device
	cuda.memcpy_htod_async(d_input, img, stream)
	#execute model
	#import pdb; pdb.set_trace();
	start = time.time()
	context.enqueue(1, bindings, stream.handle, None)
	#transfer predictions back
	cuda.memcpy_dtoh_async(output, d_output, stream)
	#syncronize threads
	stream.synchronize()
	np.save(output_npy, output)
	#heatmap = output[ :19 * 46 * 54]
	#paf =     output[ 19 * 46 * 54 : ]
	#np.save('heatmap_p2.npy', heatmap)
	#np.save('paf_p2.npy', paf)
	return output


#if __name__ == "__main__":
#   create_engine("mobilepose.engine", "./models/graph/mobilenet_thin_432x368/graph_opt.pb",  57, 46, 54, 'concat_stage7.npy', 'Openpose/concat_stage7')
#heatmap = output[:19 * 46 * 46].reshape((19, 46, 46))
#paf = output[19*46*46:].reshape((38, 46, 46))
#
#np.save('heatmap_p2.npy', heatmap)
#np.save('paf_p2.npy', heatmap)
#
#end = time.time()
#print("single image inference time {}s".format(end-start))
