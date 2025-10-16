""" PackedCaptioningSample """

from dataclasses import dataclass
from typing import List, Optional, Union
from megatron.energon.flavors.base_dataset import Sample
import torch

@dataclass
class PackedCaptioningSample(Sample):
    """Sample type for packed captioning."""
    # sample_id: str
    images: Union[
        str,
        torch.Tensor,
        List[str],
        List[List[str]],
        List[torch.Tensor],
        List[List[torch.Tensor]]
    ]
    prompts: Union[
        Optional[List[str]],
        List[List[str]]
    ]
    captions: Union[
        List[str],
        List[List[str]]
    ]