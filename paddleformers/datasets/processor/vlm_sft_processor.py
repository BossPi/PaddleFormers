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

    template = Ernie45VLTemplate()
    vision_processor = ErnieVisionProcessor(data_args=None)
    processor = SupervisedDatasetProcessor(
        template=template,
        tokenizer=None,
        processor=vision_processor,
        data_args=None,
    )
    dataset = [{"text": "This is a test."}]
    print("Input:")
    pprint(dataset)
    print("\nOutput:")
    pprint(processor.preprocess_dataset(dataset))