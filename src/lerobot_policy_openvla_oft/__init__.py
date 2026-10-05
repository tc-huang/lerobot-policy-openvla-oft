"""OpenVLA-OFT policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError as err:
    raise ImportError("lerobot is not installed. Please install lerobot to use this policy package.") from err

from .configuration_openvla_oft import OpenVLAOFTConfig
from .modeling_openvla_oft import OpenVLAOFTPolicy
from .processor_openvla_oft import make_openvla_oft_pre_post_processors

__all__ = [
    "OpenVLAOFTConfig",
    "OpenVLAOFTPolicy",
    "make_openvla_oft_pre_post_processors",
]
