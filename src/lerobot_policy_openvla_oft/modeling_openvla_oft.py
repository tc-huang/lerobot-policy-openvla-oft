from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any

import torch
from accelerate import init_empty_weights
from lerobot.policies import PreTrainedPolicy
from lerobot.policies.utils import log_model_loading_keys
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.device_utils import resolve_safetensors_device
from safetensors.torch import load_file
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


_SKIP_WEIGHT_INIT = ContextVar("openvla_oft_skip_weight_init", default=False)


@contextmanager
def skip_weight_init() -> Iterator[None]:
    """Builds policies with parameters on the meta device, for weights about to be loaded.

    Buffers are still created normally, because checkpoints do not contain them.
    """
    token = _SKIP_WEIGHT_INIT.set(True)
    try:
        yield
    finally:
        _SKIP_WEIGHT_INIT.reset(token)


def load_weights(module: nn.Module, state_dict: dict[str, Tensor], strict: bool = True) -> list[str]:
    """Loads `state_dict` into `module` by adopting its tensors instead of copying them.

    Tensors are cast to the dtype of the parameters they replace. Parameters the
    state dict does not provide, such as the action head of the base OpenVLA model,
    get PyTorch's default initialization. Returns the names of those parameters.
    """
    expected = module.state_dict()
    state_dict = {k: v.to(expected[k].dtype) for k, v in state_dict.items() if k in expected}
    missing = module.load_state_dict(state_dict, strict=strict, assign=True).missing_keys
    device = next(iter(state_dict.values())).device if state_dict else torch.device("cpu")
    for submodule in module.modules():
        if any(p.is_meta for p in submodule.parameters(recurse=False)):
            submodule.to_empty(device=device, recurse=False)
            submodule.reset_parameters()
    return missing


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
        with init_empty_weights(include_buffers=False) if _SKIP_WEIGHT_INIT.get() else nullcontext():
            self.model = build_model(config)
        cast_parameters(self.model, self.dtype)
        if config.proprio_projector_fp32 and self.model.proprio_projector is not None:
            cast_parameters(self.model.proprio_projector, torch.float32)
        self.reset()

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "OpenVLAOFTPolicy":
        """LeRobot's `from_pretrained`, without first randomly initializing the weights it loads."""
        with skip_weight_init():
            return super().from_pretrained(*args, **kwargs)

    @classmethod
    def _load_as_safetensor(
        cls, model: "OpenVLAOFTPolicy", model_file: str, map_location: str, strict: bool
    ) -> "OpenVLAOFTPolicy":
        state_dict = load_file(model_file, device=resolve_safetensors_device(map_location))
        missing = load_weights(model, state_dict, strict=strict)
        log_model_loading_keys(missing, sorted(state_dict.keys() - model.state_dict().keys()))
        return model

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
