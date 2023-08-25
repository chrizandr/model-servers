"""
ModelHandler defines a custom model handler.
"""

from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import PredictionException
import pdb

import torch
import os


def limit_memory(value=None, gpu_id=0):
    if value is not None:
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        # total_memory = 16935682048
        memory_fraction = min(1, float(value * 1024 * 1024) / total_memory)
        print(f"[HANDLER] Torch version: {torch.__version__}, Model requires {memory_fraction * 100}% GPU memory on device:{gpu_id}. Total RAM available to model: {value}MB")

        torch.cuda.set_per_process_memory_fraction(memory_fraction, gpu_id)

        try:
            torch.empty(int(total_memory * memory_fraction * 1.1), dtype=torch.int8, device=f'cuda:{gpu_id}')
            print(f"[HANDLER] Memory not limited, able to allocate {total_memory * memory_fraction * 1.1 / 1024 / 1024}MB")
        except torch.cuda.OutOfMemoryError as e:
            print(f"[HANDLER] Memory limited, fialed to allocate {total_memory * memory_fraction * 1.1 / 1024 / 1024}MB")

        torch.cuda.empty_cache()


def model_factory(init_function, *args, **kwds):
    class Model:
        """
        CAI model template
        """
        def __init__(self, device, checkpoint=None) -> None:
            self.device = device
            self.checkpoint = checkpoint
            self.init_model(device, checkpoint)

        def init_model(self, device, checkpoint):
            """
            Initialize model. This will be called during model loading time
            :param context: Initial context contains model server system properties.
            :return:
            """
            # OVERRIDE self.model with a torch.nn.module
            self.model = init_function(*args, **kwds)
            if checkpoint:
                self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            self.model.eval()
            self.model.to(device)

        def __call__(self, data, *params, **kwargs):
            with torch.no_grad():
                return self.model(data, *params, **kwargs)

    return Model


def handler_factory(init_function, preprocess_function=lambda x: x, postprocess_function=lambda x: x,
                    install_packages=None, unpack_dependencies=None, *args, **kwargs):
    class Handler(BaseHandler):
        """
        CAI model handler implementation.
        """
        import jsonpickle

        def __init__(self, *base_args):
            super(CAIHandler, self).__init__(*base_args)

            self._context = None
            self.initialized = False
            self.explain = False
            self.target = 0
            self.model_class = model_factory(init_function, *args, **kwargs)

        def install_packages(self):
            """
            Add installation of any non-pip packages here [wheels, setup.py, etc]
            """
            # OVERRIDE: System commands below need to be overridden
            # os.system("echo")
            if install_packages:
                install_packages()

        def unpack_dependencies(self):
            """
            Unpack any dependencies of the
            """
            if unpack_dependencies:
                unpack_dependencies()

        def initialize(self, context):
            """
            Initialize model. This will be called during model loading time
            :param context: Initial context contains model server system properties.
            :return:
            """
            self.install_packages()
            self.unpack_dependencies()
            self._context = context
            self.manifest = context.manifest

            self.properties = context.system_properties
            self.model_config = context.model_yaml_config
            self.model_dir = self.properties.get("model_dir")
            self.gpu_id = self.properties.get("gpu_id")
            if self.model_config is not None:
                value = self.model_config.get("memory", {}).get("maxGPUMemory")
                limit_memory(value, self.gpu_id)

            self.device = torch.device("cuda:" + str(self.gpu_id)
                                       if torch.cuda.is_available() else "cpu")

            # Read model serialize/pt file
            serialized_file = self.manifest['model']['serializedFile']
            model_pt_path = os.path.join(self.model_dir, serialized_file)

            # In case the model checkpoint is empty and weights are loaded in the init_function
            if os.path.getsize(model_pt_path) == 0 or not os.path.isfile(model_pt_path):
                model_pt_path = None

            self.model = self.model_class(self.device, model_pt_path)
            self.initialized = True

        def _decode(self, data):
            decoded_output = self.jsonpickle.decode(data)
            return decoded_output

        def _encode(self, data):
            encoded_output = self.jsonpickle.encode(data)
            return encoded_output

        def preprocess(self, data):
            """
            Transform raw input into model input data.
            :param data: list of raw requests, should match batch size
            :return: list of preprocessed model input data
            """
            # Take the input data and make it inference ready
            print(f"----------Batch size: {len(data)} ------------")

            input_data = [self._decode(x.get("data")) for x in data]
            preprocessed_data = [preprocess_function(x) for x in input_data]
            preprocessed_data = torch.stack(preprocessed_data)
            return preprocessed_data

        def inference(self, model_input):
            """
            Internal inference methods
            :param model_input: transformed model input data
            :return: list of inference output in NDArray
            """
            # Do some inference call to engine here and return output
            model_output = self.model(model_input.to(self.device))
            return model_output

        def postprocess(self, inference_output):
            """
            Return inference result.
            :param inference_output: list of inference output
            :return: list of predict results
            """
            # Take output from network and post-process to desired format
            inference_output = inference_output.detach().cpu()
            postprocess_output = [{"output": self._encode(postprocess_function(x))} for x in inference_output]
            # Add model postprocessing here
            return postprocess_output

        def handle(self, data, context):
            """
            Invoke by TorchServe for prediction request.
            Do pre-processing of data, prediction using model and postprocessing of prediciton output
            :param data: Input data for prediction
            :param context: Initial context contains model server system properties.
            :return: prediction output
            """
            model_input = self.preprocess(data)
            model_output = self.inference(model_input)
            processed_output =  self.postprocess(model_output)

            return processed_output

    return Handler

from test_model import init_function, install_packages, postprocess, preprocess, unpack_dependencies

Handler = handler_factory(init_function=init_function, preprocess_function=preprocess, postprocess_function=postprocess,
                          install_packages=install_packages, unpack_dependencies=unpack_dependencies,
                          resnet_layers=18)