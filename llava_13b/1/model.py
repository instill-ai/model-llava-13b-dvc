# pylint: skip-file
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


import traceback

import io
import time
import json
import base64
from pathlib import Path
from PIL import Image

import numpy as np
from transformers import AutoTokenizer
import torch

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils

from llava.utils import disable_torch_init
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN
)
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
    load_image_from_base64
)

class TritonPythonModel:
    # Reference: https://docs.nvidia.com/launchpad/data-science/sentiment/latest/sentiment-triton-overview.html
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
        Both keys and values are strings. The dictionary keys and values are:
        * model_config: A JSON string containing the model configuration
        * model_instance_kind: A string containing model instance kind
        * model_instance_device_id: A string containing model instance device ID
        * model_repository: Model repository path
        * model_version: Model version
        * model_name: Model name
        """
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])

        # Load the model
        model_path = str(Path(__file__).parent.absolute().joinpath('llava-v1.5-13b'))
        print(f'[DEBUG] load model under path: {model_path}')

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            device_map="auto",
        )

        self.model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            device_map="auto", # "cpu"
            max_memory={0: "12GB", 1: "12GB", 2: "12GB", 3: "12GB"},
            torch_dtype=torch.float16
        )

        # Get output configurations
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "text")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                prompt = str(pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0].decode("utf-8"))
                print(f'[DEBUG] input `prompt` type({type(prompt)}): {prompt}')

                # TODO: check model backend send in which format
                prompt_image = pb_utils.get_input_tensor_by_name(request, "prompt_image").as_numpy()[0]
                print(f'[DEBUG] input `prompt_image` type({type(prompt_image)}): {len(prompt_image)}')

                extra_params_str = str(pb_utils.get_input_tensor_by_name(request, "extra_params").as_numpy()[0].decode("utf-8"))
                # TODO: pb_utils.get_input_tensor_by_name(request, "extra_params") would be none, handle it?
                # extra_params_str = str(pb_utils.get_input_tensor_by_name(request, "extra_params").as_numpy()[0].decode("utf-8"))
                # AttributeError: 'NoneType' object has no attribute 'as_numpy'
                print(f'[DEBUG] input `extra_params` type({type(extra_params_str)}): {extra_params_str}')

                extra_params = {}
                # TODO: Add a function handle penalty
                try:
                    extra_params = json.loads(extra_params_str)
                except json.decoder.JSONDecodeError:
                    pass


                max_new_tokens = int(pb_utils.get_input_tensor_by_name(request, "max_new_tokens").as_numpy()[0])
                print(f'[DEBUG] input `max_new_tokens` type({type(max_new_tokens)}): {max_new_tokens}')

                top_k = int(pb_utils.get_input_tensor_by_name(request, "top_k").as_numpy()[0])
                print(f'[DEBUG] input `top_k` type({type(top_k)}): {top_k}')

                temperature = float(pb_utils.get_input_tensor_by_name(request, "temperature").as_numpy()[0])
                print(f'[DEBUG] input `temperature` type({type(temperature)}): {temperature}')

                random_seed = int(pb_utils.get_input_tensor_by_name(request, "random_seed").as_numpy()[0])
                print(f'[DEBUG] input `random_seed` type({type(random_seed)}): {random_seed}')

                if random_seed > 0:
                   random.seed(random_seed)
                   np.random.seed(random_seed)
                   torch.manual_seed(random_seed)
                   if torch.cuda.is_available():
                       torch.cuda.manual_seed_all(random_seed)

                stop_words = pb_utils.get_input_tensor_by_name(request, "stop_words").as_numpy()
                print(f'[DEBUG] input `stop_words` type({type(stop_words)}): {stop_words}')
                if len(stop_words) == 0:
                    stop_words = None
                elif stop_words.shape[0] > 1:
                    # TODO: Check wether shoule we decode this words
                    stop_words = list(stop_words)
                else:
                    stop_words = [str(stop_words[0])]


                vision_tower = self.model.get_vision_tower()
                if not vision_tower.is_loaded:
                    vision_tower.load_model()
                vision_tower = vision_tower.to(device='cuda', dtype=torch.float16)
                image_processor = vision_tower.image_processor

                # Handle Conversation
                # TODO: Check if conversation_prompt in parametes, if yes, don't use conv_templates
                conv_mode = "llava_llama_2"
                conv = conv_templates[conv_mode].copy()
                roles = conv.roles

                # Handle Image
                # TODO: Option for only-text input
                # TODO: Check wether protobuf satisfy the format
                # TODO: Should we resize the image?
                image = None
                image_data = base64.b64decode(prompt_image)
                image = Image.open(io.BytesIO(image_data))

                image_tensor = process_images([image], image_processor, {"image_aspect_ratio": 'pad'})
                if type(image_tensor) is list:
                    image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
                else:
                    image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

                if image is not None:
                    inp = prompt
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                    conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)

                input_ids = tokenizer_image_token(
                    prompt,
                    self.tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors='pt'
                ).unsqueeze(0).cuda()

                if stop_words is not None:
                    stopping_criteria = KeywordsStoppingCriteria(stop_words, self.tokenizer, input_ids)
                    extra_params['stopping_criteria'] = [stopping_criteria]
                # Inference
                # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
                # https://huggingface.co/docs/transformers/v4.30.1/en/main_classes/text_generation#transformers.GenerationConfig
                t0 = time.time() # calculate time cost in following function call
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    **extra_params
                )
                self.logger.log_info(f'Inference time cost {time.time()-t0}s with input lenth {len(prompt)}')
                outputs = self.tokenizer.decode(
                    output_ids[0, input_ids.shape[1]:],
                    skip_special_tokens = True
                ).strip()
                print(type(outputs))
                print('-'*100, "\n[DEBUG]", {"prompt": prompt, "outputs": outputs.encode('utf-8')}, "\n")

                text_outputs = [outputs, ]
                triton_output_tensor = pb_utils.Tensor(
                    "text", np.asarray(text_outputs, dtype=self.output0_dtype)
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[triton_output_tensor]))

            except Exception as e:
                self.logger.log_info(f"Error generating stream: {e}")
                print("DEBUG\n", traceback.format_exc())

                error = pb_utils.TritonError(f"Error generating stream: {e}")
                triton_output_tensor = pb_utils.Tensor(
                    "text", np.asarray(["N/A"], dtype=self.output0_dtype)
                )
                response = pb_utils.InferenceResponse(
                    output_tensors=[triton_output_tensor], error=error
                )
                responses.append(response)
                self.logger.log_info("The model did not receive the expected inputs")
                raise e
            return responses

    def finalize(self):
        self.logger.log_info("Cleaning up ...")
