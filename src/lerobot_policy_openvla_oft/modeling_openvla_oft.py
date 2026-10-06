from collections import deque
from typing import Any

import torch
from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from torch import Tensor, nn

from .configuration_openvla_oft import OpenVLAOFTConfig
from .image_crop import crop_and_resize, crop_boxes
from .language_model import BidirectionalLlama, openvla_llama_config
from .model import OpenVLAOFT
from .vision_backbone import FusedVisionBackbone

# Every linear layer of the pretrained VLA: the ViT attention and MLP layers, the
# vision projector, and the Llama attention and MLP projections. Matches the
# original "all-linear" LoRA, minus the layers this port removed.
LORA_TARGET_MODULES = (
    r"model\.(vision\..*\.(attn\.(qkv|proj)|mlp\.(fc1|fc2))"
    r"|vision_projector\.fc[123]"
    r"|llm\..*\.(q|k|v|o|gate|up|down)_proj)"
)


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


def cast_parameters(module: nn.Module, dtype: torch.dtype) -> None:
    """Casts the parameters of `module` to `dtype`, leaving buffers unchanged.

    `Module.to(dtype)` would also cast buffers, turning Llama's float32 RoPE
    frequencies into bfloat16, which the original never does.
    """
    for param in module.parameters():
        param.data = param.data.to(dtype)


class OpenVLAOFTPolicy(PreTrainedPolicy):
    """LeRobot policy wrapper around the OpenVLA-OFT network.

    Expects batches already processed by the OpenVLA-OFT preprocessor: images resized
    to `config.image_size` with values in [0, 1], normalized state and actions, and
    the tokenized prompt. Image cropping depends on the training mode, so it happens
    here rather than in the preprocessor.
    """

    config_class = OpenVLAOFTConfig
    name = "openvla_oft"

    def __init__(self, config: OpenVLAOFTConfig, dataset_stats: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config
        self.model = build_model(config)
        cast_parameters(self.model, self.dtype)
        if config.proprio_projector_fp32 and self.model.proprio_projector is not None:
            cast_parameters(self.model.proprio_projector, torch.float32)
        self.reset()

    @property
    def dtype(self) -> torch.dtype:
        return getattr(torch, self.config.dtype)

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> list[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def _get_default_peft_targets(self) -> dict[str, Any]:
        """LoRA defaults of the original recipe for `lerobot-train --peft.*`.

        LoRA goes on every linear layer of the pretrained VLA, while the modules that
        OpenVLA-OFT adds are trained in full. LeRobot's CLI always passes `--peft.r`
        (default 16), so the original rank 32 has to be set there.
        """
        new_modules = ("action_head", "proprio_projector")
        return {
            "target_modules": LORA_TARGET_MODULES,
            "modules_to_save": [name for name in new_modules if getattr(self.model, name) is not None],
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "init_lora_weights": "gaussian",
        }

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
        images = self._crop(torch.stack([batch[key] for key in self.config.image_features], dim=1))
        state = batch[OBS_STATE] if self.model.proprio_projector is not None else None
        with torch.autocast(images.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            return self.model(
                images,
                batch[OBS_LANGUAGE_TOKENS],
                batch[OBS_LANGUAGE_ATTENTION_MASK].bool(),
                state,
            )

    def _crop(self, images: Tensor) -> Tensor:
        """Crops each of the (B, num_images, 3, H, W) images: randomly in training, centered otherwise."""
        if self.config.image_crop_scale == 1.0:
            return images
        flat = images.flatten(0, 1)
        boxes = crop_boxes(flat.shape[0], self.config.image_crop_scale, self.training, flat.device)
        return crop_and_resize(flat, boxes).unflatten(0, images.shape[:2])
