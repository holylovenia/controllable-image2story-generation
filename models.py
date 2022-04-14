from transformers import (
    GPT2LMHeadModel,
    PreTrainedModel,
)
from typing import Optional, Tuple

import torch
import torch.nn as nn


class MLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class ClipCaptionModel(nn.Module):
    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512):
        super().__init__()
        self.prefix_length = prefix_length
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2")
        self.decoder.config.update({"n_positions": 2048})
        self.decoder_embedding_size = self.decoder.transformer.wte.weight.shape[1]
        self.mapping_network = MLP((prefix_size, (self.decoder_embedding_size * prefix_length) // 2,
                                     self.decoder_embedding_size * prefix_length))

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def get_prefix_projections(self, prefixes):
        return self.mapping_network(prefixes).view(-1, self.prefix_length, self.decoder_embedding_size)

    def get_text_embeddings(self, input_ids):
        return self.decoder.transformer.wte(input_ids)

    def forward(self, input_ids, masks, prefixes, labels: Optional[torch.Tensor] = None):
        text_embeddings = self.get_text_embeddings(input_ids)
        prefix_projections = self.get_prefix_projections(prefixes)
        caption_embeddings = torch.cat((prefix_projections, text_embeddings), dim=1)

        if labels is not None:
            dummy_token = self.get_dummy_token(input_ids.shape[0], input_ids.device)
            labels = torch.cat((dummy_token, input_ids), dim=1)
        out = self.decoder(inputs_embeds=caption_embeddings, labels=labels, attention_mask=masks)

        return out

    # def gradient_checkpointing_enable(self):
    #     self._set_gradient_checkpointing = True

    # def gradient_checkpointing_disable(self):
    #     self._set_gradient_checkpointing = False

class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.mapping_network.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        self.decoder.eval()
        return self