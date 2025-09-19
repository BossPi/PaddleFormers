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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from ...hparams import DataArguments
    
@dataclass
class VisionProcessor(ABC):
    r"""A class for vision processors."""

    data_args: "DataArguments"

    @abstractmethod
    def __call__(self, messages: list[dict], images: list[str], videos: list[str], tokenizer: "PreTrainedTokenizer") -> (dict, dict):
        r"""Process vision input."""
        ...


@dataclass
class ErnieVisionProcessor(VisionProcessor):
    r"""A processor for ERNIE vision models."""

    def __call__(self, messages: list[dict], images: list[str], videos: list[str], tokenizer: "PreTrainedTokenizer") -> (dict, dict):
        r"""Process vision input."""
        if len(images) > 0:
            image_inputs = {"image": images}
        else:
            image_inputs = {}

        if len(videos) > 0:
            video_inputs = {"video": videos}
        else:
            video_inputs = {}

        return image_inputs, video_inputs