from collections import deque
from typing import Any

import torch
from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from torch import Tensor

from .configuration_openvla_oft import OpenVLAOFTConfig
from .language_model import BidirectionalLlama, openvla_llama_config
from .model import OpenVLAOFT
from .vision_backbone import FusedVisionBackbone


def build_model(config: OpenVLAOFTConfig) -> OpenVLAOFT:
    """Builds the full-size OpenVLA-OFT network described by `config`."""
    state = config.robot_state_feature
    return OpenVLAOFT(
        vision=FusedVisionBackbone(config.image_size),
        llm=BidirectionalLlama(openvla_llama_config()),
        chunk_size=config.chunk_size,
        action_dim=config.action_feature.shape[0],
        proprio_dim=state.shape[0] if state is not None else None,
    )


class OpenVLAOFTPolicy(PreTrainedPolicy):
    """LeRobot policy wrapper around the OpenVLA-OFT network.

    Expects batches already processed by the OpenVLA-OFT preprocessor: images resized
    to `config.image_size` with values in [0, 1], normalized state and actions, and
    the tokenized prompt.
    """

    config_class = OpenVLAOFTConfig
    name = "openvla_oft"

    def __init__(self, config: OpenVLAOFTConfig, dataset_stats: dict[str, Any] | None = None):
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config
        self.model = build_model(config)
        self.model.to(self.dtype)
        if config.proprio_projector_fp32 and self.model.proprio_projector is not None:
            self.model.proprio_projector.float()
        self.reset()

    @property
    def dtype(self) -> torch.dtype:
        return getattr(torch, self.config.dtype)

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> list[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Computes the mean L1 loss between predicted and target normalized action chunks."""
        predicted = self._predict(batch)
        errors = (predicted - batch[ACTION].to(predicted.dtype)).abs()
        if self.config.mask_padded_actions:
            errors = errors[~batch[f"{ACTION}_is_pad"]]
        loss = errors.mean()
        return loss, {"l1_loss": loss.item()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        """Returns the normalized action chunk (B, chunk_size, action_dim) in float32."""
        return self._predict(batch).float()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        """Returns the next normalized action (B, action_dim), predicting a new chunk when needed."""
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()

    def _predict(self, batch: dict[str, Tensor]) -> Tensor:
        images = torch.stack([batch[key] for key in self.config.image_features], dim=1)
        state = batch[OBS_STATE] if self.model.proprio_projector is not None else None
        with torch.autocast(images.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            return self.model(
                images,
                batch[OBS_LANGUAGE_TOKENS],
                batch[OBS_LANGUAGE_ATTENTION_MASK].bool(),
                state,
            )
