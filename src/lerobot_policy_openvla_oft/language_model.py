from typing import Any

import torch
from torch import Tensor, nn
from transformers import LlamaConfig, LlamaModel

OPENVLA_VOCAB_SIZE = 32064
OPENVLA_PAD_TOKEN_ID = 32000


def openvla_llama_config(**overrides: Any) -> LlamaConfig:
    """Returns the Llama-2 7B configuration used by OpenVLA.

    OpenVLA's released config only sets the vocabulary and pad token, so every
    other field, including `rms_norm_eps=1e-6`, is a `LlamaConfig` default.
    """
    kwargs = {
        "vocab_size": OPENVLA_VOCAB_SIZE,
        "pad_token_id": OPENVLA_PAD_TOKEN_ID,
        "attn_implementation": "sdpa",
    }
    return LlamaConfig(**(kwargs | overrides))


def bidirectional_attention_mask(padding_mask: Tensor, dtype: torch.dtype) -> Tensor:
    """Builds an additive (B, 1, L, L) mask that lets every token attend to every non-padding token.

    Args:
        padding_mask: (B, L) boolean tensor, True for real tokens and False for padding.
        dtype: Dtype of the attention scores the mask is added to.
    """
    blocked = ~padding_mask[:, None, None, :]
    mask = torch.zeros(blocked.shape, dtype=dtype, device=padding_mask.device)
    return mask.masked_fill(blocked, torch.finfo(dtype).min).expand(-1, -1, padding_mask.shape[1], -1)


class BidirectionalLlama(nn.Module):
    """Llama decoder run with bidirectional instead of causal self-attention.

    Parallel decoding needs every action position to see every other one, so the
    causal mask is replaced by one that only hides padding. The language-model head
    is omitted because OpenVLA-OFT reads the final hidden states directly.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.model = LlamaModel(config)

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def eos_token_id(self) -> int:
        return self.model.config.eos_token_id

    def embed(self, input_ids: Tensor) -> Tensor:
        """Maps token ids (B, L) to embeddings (B, L, hidden_size)."""
        return self.model.embed_tokens(input_ids)

    def forward(self, inputs_embeds: Tensor, padding_mask: Tensor) -> Tensor:
        """Returns the final, normalized hidden states (B, L, hidden_size)."""
        attention_mask = bidirectional_attention_mask(padding_mask, inputs_embeds.dtype)
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
