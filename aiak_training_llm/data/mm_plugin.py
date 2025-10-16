"""
Copyright 2024 the LlamaFactory team.
Copyright (c) 2024, AIAK team. All rights reserved.
This code was adopted from https://github.com/hiyouga/LLaMA-Factory
and the source code is licensed under the Apache license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from copy import deepcopy
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, TypedDict, Union, Type

import numpy as np
from transformers.image_utils import get_image_size, to_numpy_array
from typing_extensions import override

from aiak_training_llm.utils.constants import Placeholder

from PIL import Image
from PIL.Image import Image as ImageObject


if TYPE_CHECKING:
    import torch
    from transformers.image_processing_utils import BaseImageProcessor
    class EncodedImage(TypedDict):
        """ Encoded image type. """
        path: Optional[str]
        bytes: Optional[bytes]

    ImageInput = Union[str, EncodedImage, ImageObject]
    VideoInput = str


class MMPlugin:
    """ MM Plugin """
    def __init__(self, image_token: Optional[str], video_token: Optional[str]) -> None:
        self.image_token = image_token
        self.video_token = video_token

    def _validate_input(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
    ) -> None:
        r"""
        Validates if this model accepts the input modalities.
        """
        if len(images) != 0 and self.image_token is None:
            raise ValueError("This model does not support image input.")

        if len(videos) != 0 and self.video_token is None:
            raise ValueError("This model does not support video input.")

    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        r"""
        Pre-processes a single image using min_pixels and max_pixels parameters.
        """
        # Get min_pixels and max_pixels from kwargs or use defaults
        min_pixels = kwargs.get("min_pixels", 4 * 28 * 28)  # Default: 4 * 28 * 28
        max_pixels = kwargs.get("max_pixels", 16384 * 28 * 28)  # Default: 16384 * 28 * 28

        # Convert to numpy array for easier calculation
        img_array = np.array(image)
        current_pixels = img_array.shape[0] * img_array.shape[1]

        # If image is too small, scale it up to meet min_pixels requirement
        if current_pixels < min_pixels:
            scale_factor = (min_pixels / current_pixels) ** 0.5
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            image = image.resize((new_width, new_height), resample=Image.BICUBIC)

        # If image is too large, scale it down to meet max_pixels requirement
        elif current_pixels > max_pixels:
            scale_factor = (max_pixels / current_pixels) ** 0.5
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            image = image.resize((new_width, new_height), resample=Image.NEAREST)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _get_video_sample_frames(self, video_stream: "Stream", **kwargs) -> int:
        r"""
        Computes video sample frames according to fps.
        """
        video_fps: float = kwargs.get("video_fps")
        video_maxlen: int = kwargs.get("video_maxlen")
        total_frames = video_stream.frames
        sample_frames = float(video_stream.duration * video_stream.time_base) * video_fps
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        return math.floor(sample_frames)

    def _regularize_images(self, images: Sequence["ImageInput"], **kwargs) -> List["ImageObject"]:
        r"""
        Regularizes images to avoid error. Including reading and pre-processing.
        """
        results = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])

            if not isinstance(image, ImageObject):
                raise ValueError("Expect input is a list of Images, but got {}.".format(type(image)))

            results.append(self._preprocess_image(image, **kwargs))

        return results


    def _get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
    ) -> Dict[str, "torch.Tensor"]:
        r"""
        Processes visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height

        It holds num_patches == torch.prod(image_grid_thw)
        """
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        video_processor: "BaseImageProcessor" = getattr(processor, "video_processor", image_processor)
        input_dict = {"images": None}  # default key

        if len(images) != 0:
            # Get min_pixels and max_pixels from processor if available, otherwise use defaults
            min_pixels = getattr(processor.image_processor, "min_pixels", None)
            max_pixels = getattr(processor.image_processor, "max_pixels", None)
            assert min_pixels is not None and max_pixels is not None, f'min_pixels: {min_pixels}, max_pixels: {max_pixels}'

            images = self._regularize_images(
                images,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            input_dict["images"] = images

        if len(videos) != 0:
            input_dict["videos"] = videos

        mm_inputs = {}
        if image_processor != video_processor:
            if input_dict.get("images") is not None:
                mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))
            if input_dict.get("videos") is not None:
                mm_inputs.update(video_processor(input_dict["videos"], return_tensors="pt"))
        elif input_dict.get("images") is not None or input_dict.get("videos") is not None:  # same processor (qwen2-vl)
            mm_inputs.update(image_processor(**input_dict, return_tensors="pt"))

        return mm_inputs

    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        r"""
        Pre-processes input messages before tokenization for VLMs.
        """
        self._validate_input(images, videos)
        return messages

    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        r"""
        Builds batched multimodal inputs for VLMs.
        """
        self._validate_input(images, videos)
        return {}


class Qwen2VLPlugin(MMPlugin):
    """ Qwen2VL plugin """
    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.NEAREST)

        return image

    @override
    def _get_video_sample_frames(self, video_stream: "Stream", **kwargs) -> int:
        sample_frames = super()._get_video_sample_frames(video_stream, **kwargs)
        sample_frames = sample_frames // 2 * 2
        return sample_frames

    @override
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        merge_length: int = getattr(image_processor, "merge_size") ** 2
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])

        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while Placeholder.IMAGE in content:
                if num_image_tokens > len(image_grid_thw):
                    raise ValueError("`len(images)` is less than the number of {} tokens.".format(Placeholder.IMAGE))

                content = content.replace(
                    Placeholder.IMAGE,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.image_token * (image_grid_thw[num_image_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_image_tokens += 1

            while Placeholder.VIDEO in content:
                if num_video_tokens > len(video_grid_thw):
                    raise ValueError("`len(videos)` is less than the number of {} tokens.".format(Placeholder.VIDEO))

                content = content.replace(
                    Placeholder.VIDEO,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.video_token * (video_grid_thw[num_video_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_video_tokens += 1

            message["content"] = content

        if len(images) != num_image_tokens:
            raise ValueError("The number of images does not match the number of {} tokens".format(Placeholder.IMAGE))

        if len(videos) != num_video_tokens:
            raise ValueError("The number of videos does not match the number of {} tokens".format(Placeholder.VIDEO))

        return messages, mm_inputs

    @override
    def get_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        imglens: Sequence[int],
        vidlens: Sequence[int],
        seqlens: Sequence[int],
        processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)