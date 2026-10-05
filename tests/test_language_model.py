import pytest
import torch
from tiny_models import TINY_LLAMA

from lerobot_policy_openvla_oft.language_model import (
    BidirectionalLlama,
    bidirectional_attention_mask,
    openvla_llama_config,
)


@pytest.fixture(params=["sdpa", "eager"])
def llm(request):
    torch.manual_seed(0)
    return BidirectionalLlama(openvla_llama_config(**TINY_LLAMA, attn_implementation=request.param)).eval()


def test_openvla_config_matches_released_checkpoint():
    config = openvla_llama_config()

    assert config.vocab_size == 32064
    assert config.pad_token_id == 32000
    assert config.hidden_size == 4096
    assert config.num_hidden_layers == 32
    assert config.rms_norm_eps == 1e-6


def test_mask_blocks_only_padding_keys():
    padding_mask = torch.tensor([[True, True, False]])

    mask = bidirectional_attention_mask(padding_mask, torch.float32)

    assert mask.shape == (1, 1, 3, 3)
    assert (mask[..., :2] == 0).all()
    assert (mask[..., 2] == torch.finfo(torch.float32).min).all()


def test_earlier_tokens_attend_to_later_tokens(llm):
    embeds = torch.randn(1, 4, 16)
    padding_mask = torch.ones(1, 4, dtype=torch.bool)
    changed = embeds.clone()
    changed[:, -1] += 1.0

    first = llm(embeds, padding_mask)[:, 0]
    first_after_change = llm(changed, padding_mask)[:, 0]

    assert not torch.allclose(first, first_after_change)


def test_padding_does_not_affect_real_tokens(llm):
    embeds = torch.randn(1, 5, 16)
    padding_mask = torch.tensor([[True, True, True, False, False]])
    changed = embeds.clone()
    changed[:, 3:] += 1.0

    torch.testing.assert_close(llm(embeds, padding_mask)[:, :3], llm(changed, padding_mask)[:, :3])


def test_has_no_language_model_head(llm):
    assert not any("lm_head" in name for name, _ in llm.named_parameters())
