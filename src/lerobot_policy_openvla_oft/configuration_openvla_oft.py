from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("openvla_oft")
@dataclass
class OpenvlaOftConfig(PreTrainedConfig):
    """Configuration class for OpenvlaOftPolicy.

    Args:
        n_obs_steps: Number of observation steps to use as input
        horizon: Action prediction horizon
        n_action_steps: Number of action steps to execute
        hidden_dim: Hidden dimension for the policy network
        # Add your policy-specific parameters here
    """

    horizon: int = 50
    n_action_steps: int = 50
    hidden_dim: int = 256

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.horizon:
            raise ValueError("n_action_steps cannot exceed horizon")
        raise NotImplementedError("Not implemented")

    def validate_features(self) -> None:
        """Validate input/output feature compatibility.

        Call this explicitly from your policy's __init__ — the base class does not.
        """
        if not self.image_features:
            raise ValueError("OpenvlaOftPolicy requires at least one image feature.")
        if self.action_feature is None:
            raise ValueError("OpenvlaOftPolicy requires 'action' in output_features.")
        raise NotImplementedError("Not implemented")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay
        )

    def get_scheduler_preset(self):
        """Return a LRSchedulerConfig from lerobot.optim, or None."""
        raise NotImplementedError("Not implemented")

    @property
    def observation_delta_indices(self) -> list[int] | None:
        """Relative timestep offsets the dataset loader provides per observation.

        Return `None` for single-frame policies. For temporal policies that consume
        multiple past or future frames, return a list of offsets, e.g. `[-20, -10, 0, 10]` for
        3 past frames at stride 10 and 1 future frame at stride 10.
        """
        raise NotImplementedError("Not implemented")

    @property
    def action_delta_indices(self) -> list[int]:
        """Relative timestep offsets for the action chunk the dataset loader returns."""
        raise NotImplementedError("Not implemented")

    @property
    def reward_delta_indices(self) -> None:
        raise NotImplementedError("Not implemented")
