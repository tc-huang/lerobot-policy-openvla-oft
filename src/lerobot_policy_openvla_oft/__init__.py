# __init__.py
"""Custom policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

from .configuration_openvla_oft import OpenvlaOftConfig
from .modeling_openvla_oft import OpenvlaOftPolicy
from .processor_openvla_oft import make_openvla_oft_pre_post_processors

__all__ = [
    "OpenvlaOftConfig",
    "OpenvlaOftPolicy",
    "make_openvla_oft_pre_post_processors",
]
