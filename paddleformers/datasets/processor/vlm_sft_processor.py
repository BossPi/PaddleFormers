# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from base_processor import DatasetProcessor


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    def encode_example(self, example: dict) -> dict:
        messages = example.get("messages", [])
        images = example.get("images", [])
        videos = example.get("videos", [])

        image_inputs, video_inputs = self.processor(messages=messages, images=images, videos=videos, tokenizer=self.tokenizer)
        model_input = self.template.encode(messages=messages, image_inputs=image_inputs, video_inputs=video_inputs, tokenizer=self.tokenizer)

        return model_input

    def preprocess_dataset(self, examples: list[dict]) -> list[dict]:
        model_inputs = []
        for example in examples:
            model_input = self.encode_example(example)
            model_inputs.append(model_input)

        return model_inputs

    def print_data_example(self, example: list[dict]) -> None:
        print("Example:", example)


if __name__ == "__main__":
    from pprint import pprint
    from paddleformers.datasets.vision_processor import ErnieVisionProcessor
    from paddleformers.datasets.template import Ernie45VLTemplate
    from paddleformers.transformers import AutoTokenizer

    template = Ernie45VLTemplate()
    tokenizer = AutoTokenizer.from_pretrained(
        "/root/paddlejob/workspace/env/output/peiziliang/baidu/paddle_internal/ernie-4_5-vl-28b-a3b-bf16-tokenizer",
        trust_remote_code=True,    
    )
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    vision_processor = ErnieVisionProcessor(data_args=None)
    processor = SupervisedDatasetProcessor(
        template=template,
        tokenizer=tokenizer,
        processor=vision_processor,
        data_args=None,
    )
    dataset = []
    data1 = {
        "messages": [{"role": "system", "content": "这是system中的内容。"}, 
                    {"role": "user", "content": "<image>你好呀！"}, 
                    {"role": "assistant", "content": "您好，很高兴为您服务！"}, 
                    {"role": "user", "content": "<video>今天天气怎么样？"}, 
                    {"role": "assistant", "content": "<think>\n这个问题我不会\n</think>\n\n不知道啊"}],
        "images": ["/root/paddlejob/workspace/env/output/peiziliang/ERNIE/examples/data/DoclingMatix/44/0.png"],
        "videos": ["/root/paddlejob/workspace/env/output/peiziliang/ERNIE/examples/data/NExTVideo/0008/2403134475.mp4"],
    }
    dataset.append(data1)
    data2 = {
        "messages": [{"role": "system", "content": "这是system中的内容。"}, 
                    {"role": "user", "content": "<image>你好呀！"}, 
                    {"role": "assistant", "content": "您好，很高兴为您服务！"}, 
                    {"role": "user", "content": "<video>今天天气怎么样？"}, 
                    {"role": "assistant", "content": "不知道啊"}],
        "images": ["/root/paddlejob/workspace/env/output/peiziliang/ERNIE/examples/data/DoclingMatix/44/0.png"],
        "videos": ["/root/paddlejob/workspace/env/output/peiziliang/ERNIE/examples/data/NExTVideo/0008/2403134475.mp4"],
    }
    dataset.append(data2)
    print("Input:")
    pprint(dataset)
    print("\nOutput:")
    dataset = processor.preprocess_dataset(dataset)
    pprint(dataset)