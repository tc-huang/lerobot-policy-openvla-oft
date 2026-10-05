from dataclasses import dataclass

import torch
from lerobot.configs import PreTrainedConfig
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("openvla_oft")
@dataclass
class OpenVLAOFTConfig(PreTrainedConfig):
    """Configuration class for OpenVLAOFTPolicy.

    Args:
        horizon: Action prediction horizon.
        n_action_steps: Number of action steps to execute.
        optimizer_lr: Learning rate for the AdamW optimizer preset.
        optimizer_weight_decay: Weight decay for the AdamW optimizer preset.
    """

    dtype: torch.dtype | None = torch.bfloat16

    horizon: int = 8
    n_action_steps: int = 8

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.horizon:
            raise ValueError("n_action_steps cannot exceed horizon")

    def validate_features(self) -> None:
        """Validates input/output feature compatibility.

        Must be called explicitly from the policy's __init__; the base class
        does not call it.
        """
        if not self.image_features:
            raise ValueError("OpenVLAOFTPolicy requires at least one image feature.")
        if self.action_feature is None:
            raise ValueError("OpenVLAOFTPolicy requires 'action' in output_features.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
