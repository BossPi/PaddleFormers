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

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

from typing_extensions import override

@dataclass
class Template:
    def encode(self, messages: list[dict], tokenizer: "PreTrainedTokenizer") -> dict:
        raise NotImplementedError()

    def valid_data(self, messages: list[dict]) -> bool:
        if not isinstance(messages, list):
            raise ValueError('messages must be a list')
        use_system = 0
        if messages[0].get("role", "") == "system":
            if messages[0].get("content", "") == "":
                raise ValueError('system message cannot be empty')
            use_system = 1
        role_list = ["assistant", "user"]
        for idx in range(use_system, len(messages)):
            if messages[idx].get("role", "") != role_list[idx % 2]:
                raise ValueError('message role in idx: {} must be {}'.format(idx, role_list[idx % 2]))
        return True


@dataclass
class Ernie45VLTemplate(Template):
    ignored_index = -100
    image_placeholder = '<image>'
    video_placeholder = '<video>'
    IDS_TYPE_FLAG = {"text": 0, "image": 1, "video": 1}
    def get_special_tokens(self, tokenizer: "PreTrainedTokenizer") -> None:
        self.image_start_token = tokenizer.special_tokens_map.get(
            "image_start_token", "<|IMAGE_START|>"
        )
        self.image_end_token = tokenizer.special_tokens_map.get(
            "image_end_token", "<|IMAGE_END|>"
        )
        self.video_start_token = tokenizer.special_tokens_map.get(
            "video_start_token", "<|VIDEO_START|>"
        )
        self.video_end_token = tokenizer.special_tokens_map.get(
            "video_end_token", "<|VIDEO_END|>"
        )
        self.im_patch_token = tokenizer.special_tokens_map.get(
            "image_placeholder", "<|IMAGE_PLACEHOLDER|>"
        )
        self.eos_token = tokenizer.special_tokens_map.get("eos_token", "</s>")
        self.sep_token = tokenizer.special_tokens_map.get("sep_token", "<|endofprompt|>")

    def is_thinking_data(self, messages: list[dict]) -> bool:
        return "<think>" in messages[-1]["content"] and "</think>" in messages[-1]["content"]

    def split_by_tags(self, text: str, tags: list[str]=[image_placeholder, video_placeholder]):
        pattern = '|'.join(map(re.escape, tags))
        parts = re.split(f'({pattern})', text)
        return [part for part in parts if part]

    @override
    def encode(self, messages: list[dict], image_inputs: dict, video_inputs: dict, tokenizer: "PreTrainedTokenizer") -> dict:
        self.valid_data(messages, image_inputs, video_inputs)
        
        history_str = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=self.is_thinking_data(messages))
        all_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)

        history_len = len(history_str)
        assert all_str[:history_len] == history_str, f"template(messages[:-1]): {history_str} should be a prefix of template(messages): {all_str}"

        response_str = all_str[history_len:]

        input_ids, labels, token_type_ids = [], [], []
        image_id, video_id = 0, 0

        self.get_special_tokens(tokenizer)
        for part in self.split_by_tags(history_str):
            if part == self.image_placeholder:
                added_text = f"Picture {image_id + 1}:" + self.image_start_token + self.im_patch_token * image_inputs["token_nums"][image_id] + self.image_end_token
                input_id = tokenizer.encode(added_text)
                token_type_ids.extend([self.IDS_TYPE_FLAG["image"]] * len(input_id))
                image_id += 1
            elif part == self.video_placeholder:
                added_text = f"Video {video_id + 1}:" + self.video_start_token + self.im_patch_token * sum(video_inputs["token_nums"][video_id]) + self.video_end_token
                input_id = tokenizer.encode(added_text)
                token_type_ids.extend([self.IDS_TYPE_FLAG["video"]] * len(input_id))
                video_id += 1
            else:
                input_id = tokenizer.encode(part)
                token_type_ids.extend([self.IDS_TYPE_FLAG["text"]] * len(input_id))
            input_ids.extend(input_id)
            labels.extend([self.ignored_index] * len(input_id))

        vocab = tokenizer.get_vocab()
        eos_token_id = vocab[self.eos_token]
        sep_token_id = vocab[self.sep_token]

        response_id = tokenizer.encode(response_str)
        input_ids.extend(response_id)
        token_type_ids.extend([self.IDS_TYPE_FLAG["text"]] * len(response_id))
        label_id = [eos_token_id if x == sep_token_id else x for x in response_id]
        labels.extend(label_id)

        model_input = {
            "input_ids": input_ids,
            "labels": labels,
            "token_type_ids": token_type_ids,
            "images": image_inputs,
            "grid_thw": [],
            "position_ids": [],
        }
        return model_input

    def valid_data(self, messages: list[dict], image_inputs: dict, video_inputs: dict) -> bool:
        super().valid_data(messages)
        image_count, video_count = 0, 0
        for msg in messages:
            if msg["role"] == "user":
                image_count += msg["content"].count(self.image_placeholder)
                video_count += msg["content"].count(self.video_placeholder)
            else:
                if self.image_placeholder in msg["content"] or self.video_placeholder in msg["content"]:
                    raise ValueError(f'{self.image_placeholder} and {self.video_placeholder} should only be used in user messages.')
        assert image_count == len(image_inputs["images"]), f'Number of image_placeholder should match number of images({len(image_inputs["images"])}), but got {image_count}'
        assert video_count == len(video_inputs["videos"]), f'Number of video_placeholder should match number of videos({len(video_inputs["videos"])}), but got {video_count}'
        return True