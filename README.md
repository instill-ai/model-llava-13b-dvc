---
Task: VisualQuestionAnswering
Tags:
  - VisualQuestionAnswering
  - Llava-13b
---

# Model-Llava-13b-dvc

🔥🔥🔥 Deploy [Llava-v1.6-13b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-13b) model on [VDP](https://github.com/instill-ai/vdp).

This repo contains Llava-v1.6-13b and Llava-v1.6-7b model in transformers format managed using [DVC](https://dvc.org/). For information about available extra parameters, please refer to the documentation on [GenerationConfig](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig) in the Transformer library.

Notes:

- Disk Space Requirements: 50G
- GPU Memory Requirements: 30G
- That some parameters is prohibited by Llava such as `frequency_penalty`

```
{
    "task_inputs": [
        {
            "visual_question_answering": {
              "prompt": "What is unusual inside this image?",
              "prompt_image_url": "https://artifacts.instill.tech/imgs/dog.jpg",
              // "prompt_image_url": "https://storage.googleapis.com/public-europe-west2-c-artifacts/imgs/sample1.png",
              "max_new_tokens": "300",
              "temperature": "0.9",
              "top_k": "30",
              "random_seed": "42",
              "extra_params": "{\"top_p\": 0.8}" 
            }
        }
    ]
}
```
