"""A tiny OpenVLA-OFT policy setup shared by the policy-level tests."""

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from tiny_models import TINY_IMAGE_SIZE, TINY_LLAMA, TINY_VIT

from lerobot_policy_openvla_oft import OpenVLAOFTConfig
from lerobot_policy_openvla_oft.language_model import BidirectionalLlama, openvla_llama_config
from lerobot_policy_openvla_oft.model import OpenVLAOFT
from lerobot_policy_openvla_oft.vision_backbone import FusedVisionBackbone

CHUNK_SIZE, ACTION_DIM, STATE_DIM, BATCH_SIZE = 4, 3, 5, 2
IMAGE_KEYS = (f"{OBS_IMAGES}.image", f"{OBS_IMAGES}.wrist_image")


def build_tiny_model(config, **llama_overrides):
    torch.manual_seed(0)
    state = config.robot_state_feature
    llm = BidirectionalLlama(openvla_llama_config(**(TINY_LLAMA | llama_overrides)))
    film_dim = llm.hidden_size if config.use_film else None
    return OpenVLAOFT(
        vision=FusedVisionBackbone(TINY_IMAGE_SIZE, film_dim=film_dim, **TINY_VIT),
        llm=llm,
        chunk_size=config.chunk_size,
        action_dim=config.action_feature.shape[0],
        proprio_dim=state.shape[0] if state is not None else None,
        film_mask_padding=config.film_mask_padding,
    )


def make_config(with_state=True, **overrides):
    image = PolicyFeature(type=FeatureType.VISUAL, shape=(3, TINY_IMAGE_SIZE, TINY_IMAGE_SIZE))
    input_features = dict.fromkeys(IMAGE_KEYS, image)
    if with_state:
        input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))
    return OpenVLAOFTConfig(
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        image_size=TINY_IMAGE_SIZE,
        device="cpu",
        **{"dtype": "float32", **overrides},
    )


def make_batch():
    prompt_mask = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
    batch = {key: torch.rand(BATCH_SIZE, 3, TINY_IMAGE_SIZE, TINY_IMAGE_SIZE) for key in IMAGE_KEYS}
    batch[OBS_STATE] = torch.randn(BATCH_SIZE, STATE_DIM)
    batch[OBS_LANGUAGE_TOKENS] = torch.randint(3, 64, prompt_mask.shape).masked_fill(~prompt_mask, 0)
    batch[OBS_LANGUAGE_TOKENS][:, 0] = 1
    batch[OBS_LANGUAGE_ATTENTION_MASK] = prompt_mask.long()
    batch[ACTION] = torch.randn(BATCH_SIZE, CHUNK_SIZE, ACTION_DIM)
    batch[f"{ACTION}_is_pad"] = torch.zeros(BATCH_SIZE, CHUNK_SIZE, dtype=torch.bool)
    return batch
