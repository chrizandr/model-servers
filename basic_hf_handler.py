"""
ModelHandler defines a custom model handler.
"""

from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import PredictionException
import pdb

import torch
import os
import logging
import time
from abc import ABC

import packaging.version
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer


from ts.handler_utils.distributed.pt_pippy import get_pipeline_driver
from ts.torch_handler.distributed.base_pippy_handler import BasePippyHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)
if packaging.version.parse(torch.__version__) >= packaging.version.parse("2.0.0"):
    logger.info("PyTorch version is 2.0.0 or greater")
else:
    logger.info(
        "PyTorch version is less than 2.0.0, initializing with meta device needs PyTorch 2.0.0 and greater"
    )


class TransformersSeqClassifierHandler(BasePippyHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the HF large model is loaded and
        partitioned into multiple stages each on one device using PiPPy.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        super().initialize(ctx)
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = self.local_rank

        model_path = ctx.model_yaml_config["handler"]["model_path"]
        seed = ctx.model_yaml_config["handler"]["manual_seed"]
        dtype_str = ctx.model_yaml_config["handler"]["dtype"]
        torch.manual_seed(seed)

        dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

        dtype = dtypes.get(dtype_str, torch.float32)
        if dtype_str not in dtypes:
            logger.info(
                f"Unsupported data type {dtype_str}, "
                "please submit a PR to support it. Falling back to fp32 now."
            )

        skip_init_start = time.perf_counter()
        with torch.device("meta"):
            self.model = LlamaForCausalLM.from_pretrained(
                model_path, use_cache=False, torch_dtype=dtype
            )
        skip_init_end = time.perf_counter()
        logger.info(
            f" init model time on meta device took {skip_init_end - skip_init_start} seconds"
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        # self.tokenizer.pad_token = self.tokenizer.eos_token

        self.prebuild_tokenizer(self.tokenizer, self.model)
        self.max_length = ctx.model_yaml_config["handler"]["max_length"]
        self.max_new_tokens = ctx.model_yaml_config["handler"]["max_new_tokens"]

        logger.info("Instantiating model Pipeline")
        pippy_compile_time_start = time.perf_counter()
        self.model = get_pipeline_driver(self.model, self.world_size, ctx)
        pippy_compile_time_end = time.perf_counter()

        logger.info(
            f" pippy compile time took {pippy_compile_time_end - pippy_compile_time_start} seconds on rank {self.local_rank}"
        )

        logger.info("Transformer model from path %s loaded successfully", model_dir)

        self.initialized = True

    def prebuild_tokenizer(self, tokenizer, model):
        origin_tokenizer_len = len(tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.padding_side = "left"
        new_tokenizer_len = len(tokenizer)
        if origin_tokenizer_len != new_tokenizer_len and model:
            print(f"resize embeddings from {origin_tokenizer_len} to {new_tokenizer_len}")
            model.resize_token_embeddings(new_tokenizer_len)

    def preprocess(self, requests):
        """
        Basic text preprocessing, based on the user's choice of application mode.
        Args:
            requests (list): A list of dictionaries with a "data" or "body" field, each
                            containing the input text to be processed.
        Returns:
            tuple: A tuple with two tensors: the batch of input ids and the batch of
                attention masks.
        """
        input_texts = [data.get("data") or data.get("body") for data in requests]
        input_ids_batch = []
        for input_text in input_texts:
            input_ids = self.encode_input_text(input_text)
            input_ids_batch.append(input_ids)
        input_ids_batch = torch.cat(input_ids_batch, dim=0).to(self.device)
        return input_ids_batch

    def encode_input_text(self, input_text):
        """
        Encodes a single input text using the tokenizer.
        Args:
            input_text (str): The input text to be encoded.
        Returns:
            tuple: A tuple with two tensors: the encoded input ids and the attention mask.
        """
        if isinstance(input_text, (bytes, bytearray)):
            input_text = input_text.decode("utf-8")
        logger.info("Received text: '%s'", input_text)
        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        return input_ids

    def inference(self, input_batch):
        """
        Predicts the class (or classes) of the received text using the serialized transformers
        checkpoint.
        Args:
            input_batch (tuple): A tuple with two tensors: the batch of input ids and the batch
                                of attention masks, as returned by the preprocess function.
        Returns:
            list: A list of strings with the predicted values for each input text in the batch.
        """
        input_ids_batch = input_batch
        input_ids_batch = input_ids_batch.to(self.device)
        outputs = self.model.generate(
            input_ids_batch,
            max_length=self.max_new_tokens,
        )
        generated_text = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        logger.info("Generated text: %s", generated_text)
        return generated_text

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output




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
    class CAIModel:
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

    return CAIModel


def handler_factory(init_function, preprocess_function=lambda x: x, postprocess_function=lambda x: x,
                    install_packages=None, unpack_dependencies=None, *args, **kwargs):
    class CAIHandler(BaseHandler):
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

    return CAIHandler

from test_model import init_function, install_packages, postprocess, preprocess, unpack_dependencies

Handler = handler_factory(init_function=init_function, preprocess_function=preprocess, postprocess_function=postprocess,
                          install_packages=install_packages, unpack_dependencies=unpack_dependencies,
                          resnet_layers=18)