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
from typing import TYPE_CHECKING, Optional, Union

from typing_extensions import override

@dataclass
class Template:
    def encode(self, messages: list[dict], tokenizer: "PreTrainedTokenizer") -> dict:
        pass


@dataclass
class Ernie45VLTemplate(Template):
    @override
    def encode(self, messages: list[dict], image_inputs: dict, video_inputs: dict, tokenizer: "PreTrainedTokenizer") -> dict:
        input_ids, labels, attention_mask, token_type_ids = [], [], [], []
        model_input = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "images": image_inputs,
            "grid_thw": [],
            "position_ids": [],
        }
        return model_input