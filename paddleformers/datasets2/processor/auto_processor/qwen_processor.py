# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

import re
import random
import math
import os
import PIL
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
from typing import TYPE_CHECKING, Optional, Union
from packaging import version

from .auto_processor import AutoProcessor

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PILImageResampling = PIL.Image.Resampling
else:
    PILImageResampling = PIL.Image

from typing_extensions import override

if TYPE_CHECKING:
    from ...hparams import DataArguments

from paddleformers.utils.log import logger

@dataclass
class Qwen2VLProcessor(AutoProcessor):
    image_placeholder = '<image>'
    video_placeholder = '<video>'

    def __init__(self, data_args: "DataArguments", **kwargs):
        super().__init__(data_args, **kwargs)

    def get_special_tokens(self, tokenizer: "PreTrainedTokenizer") -> None:
        self.image_token = tokenizer.special_tokens_map.get(
            "image_token", "<|image_pad|>"
        )
        self.video_token = tokenizer.special_tokens_map.get(
            "video_token", "<|video_pad|>"
        )
        # self.eos_token = tokenizer.special_tokens_map.get("eos_token", "</s>")
        # self.sep_token = tokenizer.special_tokens_map.get("sep_token", "<|endofprompt|>")

    def split_by_tags(self, text: str, tags: list[str]=[image_placeholder, video_placeholder]):
        pattern = '|'.join(map(re.escape, tags))
        parts = re.split(f'({pattern})', text)
        return [part for part in parts if part]

    @override
    def encode(self, messages: list[dict], image_inputs: list[dict], video_inputs: list[list[dict]], tokenizer: "PreTrainedTokenizer") -> dict:
        history_str = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        all_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        history_len = len(history_str)
        assert all_str[:history_len] == history_str, f"template(messages[:-1]): {history_str} should be a prefix of template(messages): {all_str}"

        response_str = all_str[history_len:]

        # image_inputs, video_inputs = self.process_vision_info(
        #     messages,
        #     image_inputs,
        #     video_inputs,
        #     tokenizer,
        # )
        # return None

        input_ids, labels = [], []
        # pixel_values, vision_grid_thws = [], []
        image_id, video_id = 0, 0

        self.get_special_tokens(tokenizer)
        for part in self.split_by_tags(history_str):
            if part == self.image_placeholder:
                added_text = self.image_token * image_inputs["token_nums"][image_id]
                input_id = tokenizer.encode(added_text)
                # pixel_values.append(image_inputs["images"][image_id])
                # vision_grid_thws.append(image_inputs["grid_thw"][image_id])
                image_id += 1
            elif part == self.video_placeholder:
                added_text = self.video_token * video_inputs["token_nums"][video_id]
                input_id = tokenizer.encode(added_text)
                # pixel_values.append(video_inputs["images"][video_id])
                # vision_grid_thws.append(video_inputs["grid_thw"][video_id])
                video_id += 1
            else:
                input_id = tokenizer.encode(part)
            input_ids.extend(input_id)
            # labels.extend([self.ignored_index] * len(input_id))

        return None
        
        # pixel_values = np.concatenate(
        #     pixel_values, axis=0
        # )
        # vision_grid_thws = np.array(vision_grid_thws)

        # vocab = tokenizer.get_vocab()
        # eos_token_id = vocab[self.eos_token]
        # sep_token_id = vocab[self.sep_token]

        # response_id = tokenizer.encode(response_str)
        # input_ids.extend(response_id)
        # token_type_ids.extend([self.IDS_TYPE_FLAG["text"]] * len(response_id))
        # label_id = [eos_token_id if x == sep_token_id else x for x in response_id]
        # labels.extend(label_id)

        # position_ids = self.position_ids_for_rope_3d(input_ids, vision_grid_thws, tokenizer.encode(self.im_patch_token)[0])

        # model_input = {
        #     "input_ids": input_ids,
        #     "images": pixel_values,
        #     "labels": labels,
        #     "grid_thw": vision_grid_thws,
        #     "position_ids": position_ids,
        # }
        # return model_input