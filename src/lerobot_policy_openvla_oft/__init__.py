# __init__.py
"""Custom policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

from .configuration_my_policy import MyPolicyConfig
from .modeling_my_policy import MyPolicy
from .processor_my_policy import make_my_policy_pre_post_processors

__all__ = [
    "MyPolicyConfig",
    "MyPolicy",
    "make_my_policy_pre_post_processors",
]