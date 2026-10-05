from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim import AdamWConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR


@LRSchedulerConfig.register_subclass("openvla_oft_step_decay")
@dataclass
class StepDecaySchedulerConfig(LRSchedulerConfig):
    """Constant learning rate, multiplied by `decay_factor` once after `decay_steps` steps."""

    decay_steps: int = 100_000
    decay_factor: float = 0.1
    num_warmup_steps: int | None = None

    def build(self, optimizer: Optimizer, num_training_steps: int) -> MultiStepLR:
        return MultiStepLR(optimizer, milestones=[self.decay_steps], gamma=self.decay_factor)


@PreTrainedConfig.register_subclass("openvla_oft")
@dataclass
class OpenVLAOFTConfig(PreTrainedConfig):
    """Configuration for OpenVLAOFTPolicy.

    Defaults follow the OpenVLA-OFT LIBERO recipe.

    Args:
        chunk_size: Number of future actions predicted in parallel per forward pass.
        n_action_steps: Number of actions from each chunk executed open-loop before
            querying the policy again.
        normalization_mapping: Normalization mode per feature type. State and action
            are mapped from their [q01, q99] range to [-1, 1].
        optimizer_lr: Peak learning rate.
        optimizer_weight_decay: AdamW weight decay.
        optimizer_grad_clip_norm: Gradient clipping norm; 0 disables clipping.
        scheduler_decay_steps: Step after which the learning rate is decayed.
        scheduler_decay_factor: Multiplicative learning rate decay factor.
    """

    chunk_size: int = 8
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    optimizer_lr: float = 5e-4
    optimizer_weight_decay: float = 1e-2
    optimizer_grad_clip_norm: float = 0.0

    scheduler_decay_steps: int = 100_000
    scheduler_decay_factor: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < self.n_action_steps <= self.chunk_size:
            raise ValueError(
                f"`n_action_steps` must be in [1, chunk_size={self.chunk_size}], got {self.n_action_steps}."
            )

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("OpenVLA-OFT requires at least one image feature.")
        if self.action_feature is None:
            raise ValueError("OpenVLA-OFT requires an action output feature.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> StepDecaySchedulerConfig:
        return StepDecaySchedulerConfig(
            decay_steps=self.scheduler_decay_steps, decay_factor=self.scheduler_decay_factor
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
