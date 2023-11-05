# pylint: skip-file
import traceback

import io
import time
import json
import base64
from pathlib import Path

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils

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

        print("Initializing Triton Python model...")

        # Get output configurations
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "text")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        print("Handle requests...")
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

                stop_words = pb_utils.get_input_tensor_by_name(request, "stop_words").as_numpy()
                print(f'[DEBUG] input `stop_words` type({type(stop_words)}): {stop_words}')
                if len(stop_words) == 0:
                    stop_words = None
                elif stop_words.shape[0] > 1:
                    # TODO: Check wether shoule we decode this words
                    stop_words = list(stop_words)
                else:
                    stop_words = [str(stop_words[0])]


                raise ValueError("test")
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
