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
import random
import math
from dataclasses import dataclass
from PIL import Image
from typing import TYPE_CHECKING, Optional, Union

from typing_extensions import override

from paddleformers.utils.log import logger

@dataclass
class Template:
    def __init__(self, data_args: "DataArguments"):
        """
        init
        """
        self.data_args = data_args

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
    MAX_RATIO = 200

    def __init__(self, data_args: "DataArguments"):
        super().__init__(data_args)
        self.max_seq_len = data_args.max_seq_len
        self.patch_size = data_args.patch_size
        self.merge_size = data_args.merge_size
        self.spatial_conv_size = data_args.spatial_conv_size
        self.temporal_conv_size = data_args.temporal_conv_size
        self.min_pixels = data_args.min_pixels
        self.max_pixels = data_args.max_pixels
        self.video_min_pixels = data_args.video_min_pixels
        self.video_max_pixels = data_args.video_max_pixels
        self.render_timestamp = data_args.render_timestamp

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
    
    def smart_resize(self, height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> (int, int):
        """
        Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.

        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

        3. The aspect ratio of the image is maintained as closely as possible.
        """
        def round_by_factor(number: int, factor: int) -> int:
            """Returns the closest integer to 'number' that is divisible by 'factor'."""
            return round(number / factor) * factor

        def ceil_by_factor(number: int, factor: int) -> int:
            """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
            return math.ceil(number / factor) * factor

        def floor_by_factor(number: int, factor: int) -> int:
            """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
            return math.floor(number / factor) * factor

        if max(height, width) / min(height, width) > self.MAX_RATIO:
            if height > width:
                new_width = max(factor, round_by_factor(width, factor))
                new_height = floor_by_factor(new_width * self.MAX_RATIO, factor)
            else:
                new_height = max(factor, round_by_factor(height, factor))
                new_width = floor_by_factor(new_height * self.MAX_RATIO, factor)

            logger.info(
                f"absolute aspect ratio must be smaller than {self.MAX_RATIO}, got {max(height, width) / min(height, width)},\
                resize to {max(new_height, new_width) / min(new_height, new_width)}"
            )

            height = new_height
            width = new_width

        h_bar = max(factor, round_by_factor(height, factor))
        w_bar = max(factor, round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = floor_by_factor(height / beta, factor)
            w_bar = floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, factor)
            w_bar = ceil_by_factor(width * beta, factor)

        if min_pixels > h_bar * w_bar or h_bar * w_bar > max_pixels:
            raise ValueError(f"encounter invalid h_bar: {h_bar}, w_bar: {w_bar}")

        return h_bar, w_bar

    def get_images_token_num(self, image_info: dict, min_pixels: int, max_pixels: int) -> int:
        r"""Get the number of tokens for a single image."""
        assert "image_width" in image_info and "image_height" in image_info

        resized_height, resized_width = self.smart_resize(
            image_info["image_height"],
            image_info["image_width"],
            factor=self.patch_size * self.merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        height_patches = resized_height // self.patch_size
        width_patches = resized_width // self.patch_size

        return height_patches * width_patches // (self.spatial_conv_size ** 2)

    def squeeze_video(self, messages: list[dict], image_inputs: list[dict], video_inputs: list[list[dict]], tokenizer: "PreTrainedTokenizer") -> (list[list[dict]], int):
        r"""Squeeze video into one sequence."""
        def judge_single_adaptive_resolution(
            tmp_video_min_pixels, tmp_video_max_pixels, quota_num_tokens
        ):
            """judge single resolution"""
            tmp_vision_tokens_num = 0
            for video in video_inputs:
                tmp_vision_tokens_num += self.get_images_token_num(
                    video[0],
                    tmp_video_min_pixels,
                    tmp_video_max_pixels
                ) / self.temporal_conv_size * len(video)

            if tmp_vision_tokens_num < quota_num_tokens:
                return True, tmp_vision_tokens_num

            return False, tmp_vision_tokens_num

        def judge_adaptive_resolution(
            permt_video_min_pixels, permt_video_max_pixels, quota_num_tokens
        ):
            """judge adaptive resolution"""
            left = int(permt_video_min_pixels)
            right = int(permt_video_max_pixels)
            flag = False

            try:
                while left < right:
                    mid = (left + right + 1) // 2
                    tmp_flag, permt_vision_num_tokens = (
                        judge_single_adaptive_resolution(
                            permt_video_min_pixels, mid, quota_num_tokens
                        )
                    )
                    if tmp_flag:
                        left = mid
                        flag = True
                        fi_permt_vision_num_tokens = permt_vision_num_tokens
                    else:
                        right = mid - 1
            except ValueError:
                logger.debug(
                    "[BINARY SEARCH] encounter resized shape smaller than min_pixels, early exit!"
                )
                return False, right, permt_vision_num_tokens

            if flag:
                return flag, left, fi_permt_vision_num_tokens
            return flag, left, permt_vision_num_tokens
        
        def calculate_ratios_with_min_one(numbers):
            if not numbers:
                return []

            total = sum(numbers)
            if total == 0:
                raise ValueError("the sum of numbers cannot be 0")

            base_ratios = [num / total for num in numbers]

            min_ratio = min(base_ratios)

            adjusted_ratios = [round(ratio / min_ratio) for ratio in base_ratios]

            return adjusted_ratios

        def remove_frames_for_video(video, num_frames_to_be_deleted):
            num_frames = len(video)
            max_frames = num_frames - num_frames_to_be_deleted
            frames = list(range(num_frames))

            frame_interval = num_frames // max_frames if num_frames >= max_frames else 1
            frame_indices_selected = frames[::frame_interval]
            if len(frame_indices_selected) > max_frames:
                indices_selected = random.sample(
                    range(1, len(frame_indices_selected) - 1), k=max_frames - 2
                )
                indices_selected.sort()
                indices_selected = (
                    [0] + indices_selected + [len(frame_indices_selected) - 1]
                )
                frame_indices_selected = [
                    frame_indices_selected[i] for i in indices_selected
                ]
            select_video = [video[i] for i in frame_indices_selected]
            return select_video

        video_min_pixels = self.video_min_pixels
        video_max_pixels = self.video_max_pixels

        assert video_min_pixels > 0
        assert video_max_pixels > 0

        for video in video_inputs:
            video_min_pixels = min(
                video_min_pixels,
                video[0]["image_width"] * video[0]["image_height"]
            )

        all_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)

        text_token_count = 0
        image_id, video_id = 0, 0
        self.get_special_tokens(tokenizer)
        for part in self.split_by_tags(all_str):
            if part == self.image_placeholder:
                added_text = f"Picture {image_id + 1}:" + self.image_start_token + self.image_end_token
                image_id += 1
            elif part == self.video_placeholder:
                added_text = f"Video {video_id + 1}:" + self.video_start_token + self.video_end_token
                video_id += 1
            else:
                added_text = part
            text_token_count += len(tokenizer.encode(added_text))
        
        image_token_num = 0
        for image in image_inputs:
            image_token_num += self.get_images_token_num(
                image,
                self.min_pixels,
                self.max_pixels
            )
        video_token_limit = self.max_seq_len - text_token_count - image_token_num
        video_token_num = 0
        for video in video_inputs:
            video_token_num += self.get_images_token_num(
                video[0],
                video_min_pixels,
                video_max_pixels
            ) / self.temporal_conv_size * len(video)

        if video_token_num <= video_token_limit:
            return video_inputs, video_max_pixels

        (judge_adaptive_flag, judge_video_max_pixels, permt_vision_tokens_num) = (
            judge_adaptive_resolution(
                video_min_pixels, video_max_pixels, video_token_limit
            )
        )

        logger.debug(
            f"after adjust, video_min_pixels: {video_min_pixels}, video_max_pixels: {judge_video_max_pixels}"
        )
        logger.debug(
            f"after adjust, video_token_num: {permt_vision_tokens_num}, video_token_limit: {video_token_limit}"
        )
        video_max_pixels = judge_video_max_pixels
        vision_tokens_num = permt_vision_tokens_num

        if judge_adaptive_flag:
            return video_inputs, video_max_pixels
        else:
            token_per_frame_per_video = [
                self.get_images_token_num(
                    video[0],
                    video_min_pixels,
                    video_max_pixels
                ) / self.temporal_conv_size for video in video_inputs
            ]

            tokens_to_delete = vision_tokens_num - video_token_limit

            num_frames_to_be_deleted_for_each_video = [0 for _ in video_inputs]
            video_cnt = 0
            break_cond = 0

            token_per_video = [
                tokens * len(video) for tokens, video in zip(token_per_frame_per_video, video_inputs)
            ]
            ratio = calculate_ratios_with_min_one(token_per_video)
            ratio = [i * self.temporal_conv_size for i in ratio]

            while tokens_to_delete > 0 and break_cond < len(video_inputs):
                video_index = video_cnt % len(video_inputs)
                if (
                    len(grouped_frames[video_index])
                    - num_frames_to_be_deleted_for_each_video[video_index]
                    - ratio[video_index]
                    >= 2
                ):
                    # image tokens
                    num_frames_to_be_deleted_for_each_video[video_index] += ratio[
                        video_index
                    ]
                    tokens_to_delete -= (
                        ratio[video_index] * token_per_frame_per_video[video_index]
                    )

                    break_cond = 0
                else:
                    break_cond += 1
                video_cnt += 1
            
            new_video_inputs = []
            for video, num_frames_to_be_deleted in zip(
                video_inputs, num_frames_to_be_deleted_for_each_video
            ):
                logger.debug(
                    f"original frames {len(video)}, num_frames_to_be_deleted {num_frames_to_be_deleted}, "
                    + f"final frames {len(video) - num_frames_to_be_deleted}"
                )
                new_video_inputs.extend(
                    remove_frames_for_video(
                        video,
                        num_frames_to_be_deleted
                    )
                )
            return new_video_inputs, video_max_pixels

    def convert_to_rgb(self, image_input: dict) -> Image.Image:
        def has_transparent_background(img):
            """has_transparent_background"""
            if img.mode in ("RGBA", "LA") or (
                img.mode == "P" and "transparency" in img.info
            ):
                # Check for any pixel with alpha channel less than 255 (fully opaque)
                alpha = img.convert("RGBA").split()[-1]
                if alpha.getextrema()[0] < 255:
                    return True
            return False
        
        def add_white_background(img):
            """
            Add a white background to a transparent background image
            """
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            # Create an image with a white background and the same size as the original image
            img_white_background = Image.new("RGBA", img.size, (255, 255, 255))

            # Paste the original image onto a white background
            img_white_background.paste(img, (0, 0), img)

            return img_white_background
        
        def change_I16_to_L(img):
            """
            Convert image from I;16 mode to L mode
            """
            # Since the point function in I mode only supports addition, subtraction, and multiplication, the following * (1 / 256) cannot be changed to division.
            return img.point(lambda i: i * (1 / 256)).convert("L")

        image = image_input["image"]
        try:
            if image.mode == "I;16":
                image = change_I16_to_L(image)
            if has_transparent_background(image):
                image = add_white_background(image)
        except Exception:
            pass
        return image.convert("RGB")

    def process_images(self, image_inputs: list[dict], add_timestamps: bool = False):
        r"""Process image."""
        imgs = []
        predetermined_grid_thw = []
        for image_input in image_inputs:
            image = self.convert_to_rgb(image_input)
        return {"images": [image], "token_nums": [5]}

    def process_vision_info(self, messages: list[dict], image_inputs: list[dict], video_inputs: list[list[dict]], tokenizer: "PreTrainedTokenizer") -> (list[dict], list[list[dict]]):
        r"""Process vision info."""
        print("before squeezing: ", video_inputs)
        video_inputs, video_max_pixels = self.squeeze_video(
                messages=messages,
                image_inputs=image_inputs, 
                video_inputs=video_inputs,
                tokenizer=tokenizer,
            )
        print("after squeezing: ", video_inputs)

        final_image_inputs, final_video_inputs = [], []
        for image_input in image_inputs:
            _ = self.process_images([image_input])
        
        for video_input in video_inputs:
            _ = self.process_images(video_input, add_timestamps=True)
        
        image_inputs = {"images": image_inputs, "token_nums": [10]}
        video_inputs = {"videos": video_inputs, "token_nums": [[5, 5, 5]]}
        return image_inputs, video_inputs

    @override
    def encode(self, messages: list[dict], image_inputs: list[dict], video_inputs: list[list[dict]], tokenizer: "PreTrainedTokenizer") -> dict:
        self.valid_data(messages, image_inputs, video_inputs)
        
        history_str = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=self.is_thinking_data(messages))
        all_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)

        history_len = len(history_str)
        assert all_str[:history_len] == history_str, f"template(messages[:-1]): {history_str} should be a prefix of template(messages): {all_str}"

        response_str = all_str[history_len:]

        image_inputs, video_inputs = self.process_vision_info(
            messages,
            image_inputs,
            video_inputs,
            tokenizer,
        )

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
        assert image_count == len(image_inputs), f'Number of image_placeholder should match number of images({len(image_inputs)}), but got {image_count}'
        assert video_count == len(video_inputs), f'Number of video_placeholder should match number of videos({len(video_inputs)}), but got {video_count}'
        return True