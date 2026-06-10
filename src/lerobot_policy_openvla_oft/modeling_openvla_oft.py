from typing import Any

import torch

from lerobot.policies import PreTrainedPolicy

from .configuration_openvla_oft import OpenvlaOftConfig


class OpenvlaOftPolicy(PreTrainedPolicy):
    config_class = OpenvlaOftConfig
    name = "openvla_oft"

    def __init__(self, config: OpenvlaOftConfig, dataset_stats: dict[str, Any] = None):
        super().__init__(config, dataset_stats)
        config.validate_features()  # not called automatically by the base class
        self.config = config
        self.model = None  # your nn.Module here

    def reset(self):
        """Reset per-episode state. Called by lerobot-eval at the start of each episode."""
        # ...
        raise NotImplementedError("Not implemented")

    def get_optim_params(self) -> dict:
        """Return parameters to pass to the optimizer (e.g. with per-group lr/wd)."""
        return {"params": self.parameters()}

    def predict_action_chunk(
        self, batch: dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """Return the full action chunk (B, chunk_size, action_dim) for the current observation."""
        # ...
        raise NotImplementedError("Not implemented")

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Return a single action for the current timestep (called every step at inference)."""
        # ...
        raise NotImplementedError("Not implemented")

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict | None]:
        """Compute the training loss.

        Returns `(loss, output_dict)`. `output_dict` may be `None`; everything in it must be
        logging-friendly Python natives (no tensors with gradients).

        `batch["action_is_pad"]` is a bool mask of shape (B, horizon) that marks
        timesteps padded because the episode ended before `horizon` steps; you
        can exclude those from your loss.
        """
        # ...
        # return loss, {"some_loss_component": some_loss_component.item()}
        raise NotImplementedError("Not implemented")
