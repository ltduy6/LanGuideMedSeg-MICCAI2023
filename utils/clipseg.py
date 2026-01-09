from typing import Optional

from torch import nn
import torch
from transformers import CLIPSegForImageSegmentation


class CLIPSeg(nn.Module):
    r"""CLIPSeg Official implementation from HuggingFace.

    Args:
        clipseg_hf_api (str): HuggingFace api to import the CLIPSeg implementation; Eg:'CIDAS/clipseg-rd64-refined'
        freeze_encoder (bool): Whether or not to freeze the encoders of pretrained CLIPSeg; Default is False.
        freeze_decoder (bool): Whether or not to freeze the decoder of pretrained CLIPSeg; Default is False.
    """
    def __init__(
        self,
        clipseg_hf_api: str,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
    ) -> None:
        super().__init__()

        self.clipseg = CLIPSegForImageSegmentation.from_pretrained(clipseg_hf_api)

        self.clipseg.clip.requires_grad_(not freeze_encoder)
        self.clipseg.decoder.requires_grad_(not freeze_decoder)

    def forward(
        self, 
        data
    ) -> torch.Tensor:
        pixel_values, text = data
        input_ids = text["input_ids"]
        attention_mask = text["attention_mask"]

        if pixel_values.shape[1] == 1:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)
        
        B, C, H, W = pixel_values.shape
        outputs = self.clipseg(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask
        )

        return torch.sigmoid(outputs.logits.view(B, 1, H, W))