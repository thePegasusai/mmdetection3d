import os.path
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

additional_outputs = dict(
        voxel_size=[0.2, 0.2, 16],
        pc_range=[0, -55.2],
        score_threshold=[0.1],
        outsize_factor=[4],
        post_center_limit_range=[-10, -65.2, -13, 138, 65.2, 13],
        pre_max_size=[1000],
        post_max_size=[83],
        nms_thr=[0.2],
    )


def build_trt_engine(onnx_export_pathname):
    print('TensorRT version: ', trt.__version__)
    trt_logger = trt.Logger(trt.Logger.INFO)
    def add_const_output(network, output_name, const_value):
        const_layer = network.add_constant(const_value.shape, const_value)
        const_layer.name = output_name
        const_layer.get_output(0).name = output_name
        network.mark_output(tensor=const_layer.get_output(0))

    with trt.Builder(trt_logger) as builder:
        builder.max_batch_size = 1

        builder_config = builder.create_builder_config()
        builder_config.reset()
        builder_config.max_workspace_size = 500 * 1024 * 1024
        builder_config.default_device_type = trt.DeviceType.GPU
        builder_config.engine_capability = trt.EngineCapability.STANDARD

        # Build TensorRT engine
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with builder.create_network(EXPLICIT_BATCH) as network:
            with trt.OnnxParser(network, trt_logger) as parser:
                if not parser.parse_from_file(onnx_export_pathname):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    raise ValueError(f'Failed to parse ONNX to TensorRT network')

                for key, val in additional_outputs.items():
                    add_const_output(network, key, np.array(val).astype(np.float32))

                # build engine
                network.name = os.path.basename(onnx_export_pathname).split('.')[0]
                cuda_engine = builder.build_engine(network, builder_config)

                if cuda_engine is not None:
                    with open(f'{network.name}.trt', 'wb') as f:
                        f.write(cuda_engine.serialize())
                    del cuda_engine
                else:
                    raise ValueError(f'Failed to build TensorRT engine')


def test_trt_engine(trt_engine_pathname):
    with open(trt_engine_pathname, 'rb') as f:
        with trt.Runtime(trt.Logger(trt.Logger.INFO)) as runtime:
            cuda_engine = runtime.deserialize_cuda_engine(f.read())

    with cuda_engine.create_execution_context() as context:
        # Allocate GPU memory
        bindings_mem = []
        for idx in range(cuda_engine.num_bindings):
            binding_shape = context.get_binding_shape(idx)
            binding_dtype = cuda_engine.get_binding_dtype(idx)
            binding_name = cuda_engine.get_binding_name(idx)
            is_input = cuda_engine.binding_is_input(idx)
            print(f'binding {idx}: {binding_name}{binding_shape}, {binding_dtype}, {"input" if is_input else "output"}')
            if is_input:
                gpu_mem = cuda.to_device(np.ndarray(binding_shape, dtype=np.float32))
            else:
                gpu_mem = cuda.mem_alloc(trt.volume(binding_shape) * binding_dtype.itemsize)
            bindings_mem.append(gpu_mem)

        # Execute engine
        rc = context.execute_v2(bindings=[int(binding_mem) for binding_mem in bindings_mem])
        print('execute_v2() rc:', rc)


if  __name__ == '__main__':
    # Depending on GPU utilization TensorRT might not have enough memory to build and execute engine in the same run.
    # If you get CUDA memory related error try to build and run in separate steps
    #build_trt_engine('CenterPoint.onnx')
    test_trt_engine('CenterPoint.trt')