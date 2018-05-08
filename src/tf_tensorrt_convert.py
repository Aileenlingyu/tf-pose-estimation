import tensorrt as trt
from tensorrt.lite import Engine
import time
import uff, os
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
from tensorrt.parsers import uffparser
import numpy as np

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)


def create_engine(name, model_path, height, width, input_layer='image',
                  output_layer='Openpose/concat_stage7', half16=False):
    if not os.path.exists(name):
        # Load your newly created Tensorflow frozen model and convert it to UFF
        # import pdb; pdb.set_trace();
        uff_model = uff.from_tensorflow_frozen_model(model_path, [output_layer])  # , output_filename = 'mobilepose.uff')
        dump = open(name.replace('engine', 'uff'), 'wb')
        dump.write(uff_model)
        dump.close()
        # Create a UFF parser to parse the UFF file created from your TF Frozen model
        parser = uffparser.create_uff_parser()
        parser.register_input(input_layer, (3, height, width), 0)
        parser.register_output(output_layer)

        # Build your TensorRT inference engine
        # This step performs (1) Tensor fusion (2) Reduced precision
        # (3) Target autotuning (4) Tensor memory management
        engine = trt.utils.uff_to_trt_engine(G_LOGGER,
                                             uff_model,
                                             parser,
                                             1,
                                             1 << 20,
                                             datatype=trt.infer.DataType.FLOAT if not half16 else trt.infer.DataType.HALF)
        trt.utils.write_engine_to_file(name, engine.serialize())
    else:
        engine = trt.utils.load_engine(G_LOGGER, name)

    return engine

def tensorrt_inference(img, c_o, height, width, context):
    output = np.empty(c_o * int(height/8) * int(width/8), dtype=np.float32)
    # alocate device memory
    d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
    d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, img, stream)
    # execute model
    # import pdb; pdb.set_trace();
    start = time.time()
    
    context.enqueue(1, bindings, stream.handle, None)
    #end = time.time()
    #print('prediction on single image run 10 times takes {}s'.format(end - start))
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()
    #np.save(output_npy, output)
    # heatmap = output[ :19 * 46 * 54]
    # paf =     output[ 19 * 46 * 54 : ]
    # np.save('heatmap_p2.npy', heatmap)
    # np.save('paf_p2.npy', paf)
    return output


def create_engine_from_caffe(name, model, proto, input_layer, output_layer, half16=False):
    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
    if not os.path.exists(name):
        engine = trt.utils.caffe_to_trt_engine(G_LOGGER,
                                       proto,
                                       model,
                                       1,
                                       1 << 20,
                                       [output_layer],
                                       trt.infer.DataType.FLOAT if not half16 else trt.infer.DataType.HALF)
        
        trt.utils.write_engine_to_file(name, engine.serialize())
    else:
        engine = trt.utils.load_engine(G_LOGGER, name)

    print("engine created ....")
    return engine 