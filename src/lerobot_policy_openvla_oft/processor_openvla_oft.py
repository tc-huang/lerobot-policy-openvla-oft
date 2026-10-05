from typing import Any

import torch
from lerobot.processor import PolicyAction, PolicyProcessorPipeline


def make_openvla_oft_pre_post_processors(
    config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    raise NotImplementedError
