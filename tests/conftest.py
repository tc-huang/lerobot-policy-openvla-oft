import pytest
import torch
from tiny_models import TINY_IMAGE_SIZE, TINY_LLAMA, TINY_VIT

from lerobot_policy_openvla_oft.language_model import BidirectionalLlama, openvla_llama_config
from lerobot_policy_openvla_oft.vision_backbone import FusedVisionBackbone


@pytest.fixture
def tiny_vision():
    torch.manual_seed(0)
    return FusedVisionBackbone(TINY_IMAGE_SIZE, **TINY_VIT)


@pytest.fixture
def tiny_llm():
    torch.manual_seed(0)
    return BidirectionalLlama(openvla_llama_config(**TINY_LLAMA))
