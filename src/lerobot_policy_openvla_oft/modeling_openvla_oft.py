from typing import Any

import torch
from lerobot.policies import PreTrainedPolicy

from .configuration_openvla_oft import OpenVLAOFTConfig


class OpenVLAOFTPolicy(PreTrainedPolicy):
    config_class = OpenVLAOFTConfig
    name = "openvla_oft"

    def __init__(self, config: OpenVLAOFTConfig, dataset_stats: dict[str, Any] | None = None):
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config

    def reset(self):
        """Resets per-episode state. Called by lerobot-eval at each episode start."""
        raise NotImplementedError

    def get_optim_params(self) -> dict:
        return {"params": self.parameters()}

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Returns the action chunk of shape (B, horizon, action_dim)."""
        raise NotImplementedError

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Returns a single action of shape (B, action_dim) for the current step."""
        raise NotImplementedError

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        """Computes the training loss.

        Returns:
            A `(loss, output_dict)` tuple. `output_dict` may be None and must only
            contain logging-friendly Python natives.
        """
        raise NotImplementedError
