"""Converts a released OpenVLA-OFT checkpoint into a LeRobot policy directory.

Usage:
    python -m lerobot_policy_openvla_oft.convert_checkpoint \\
        --repo-id moojink/openvla-7b-oft-finetuned-libero-spatial \\
        --output-dir outputs/checkpoints/libero-spatial
"""

import argparse
import json
import re
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from safetensors.torch import load_file
from torch import Tensor

from .configuration_openvla_oft import OpenVLAOFTConfig
from .model import OpenVLAOFT
from .modeling_openvla_oft import OpenVLAOFTPolicy
from .processor_openvla_oft import make_openvla_oft_pre_post_processors

CHECKPOINT_FILES = [
    "model.safetensors.index.json",
    "model-*.safetensors",
    "action_head--*.pt",
    "proprio_projector--*.pt",
    "dataset_statistics.json",
]

LIBERO_IMAGE_KEYS = (f"{OBS_IMAGES}.image", f"{OBS_IMAGES}.image2")

_VLA_PREFIXES = {
    "vision_backbone.featurizer.": "vision.dinov2.vit.",
    "vision_backbone.fused_featurizer.": "vision.siglip.vit.",
    "projector.": "vision_projector.",
    "language_model.model.": "llm.model.",
}
_ACTION_HEAD_LAYERS = {
    "layer_norm1": "input_norm",
    "fc1": "input_proj",
    "layer_norm2": "output_norm",
    "fc2": "output_proj",
}


def convert_vla_key(key: str) -> str | None:
    """Maps a key of the released Hugging Face model to `OpenVLAOFT`, or None to drop it."""
    if key == "language_model.lm_head.weight":
        return None
    for old, new in _VLA_PREFIXES.items():
        if key.startswith(old):
            key = new + key.removeprefix(old)
            return re.sub(r"\.(ls[12])\.scale_factor$", r".\1.gamma", key)
    raise KeyError(f"Unexpected checkpoint key: {key}")


def convert_action_head_key(key: str) -> str:
    """Maps a key of the released `L1RegressionActionHead` state dict to `OpenVLAOFT`."""
    key = key.removeprefix("module.").removeprefix("model.")
    block = re.fullmatch(r"mlp_resnet_blocks\.(\d+)\.ffn\.([01])\.(\w+)", key)
    if block:
        index, layer, param = block.groups()
        return f"action_head.blocks.{index}.{'norm' if layer == '0' else 'linear'}.{param}"
    layer, param = key.split(".")
    return f"action_head.{_ACTION_HEAD_LAYERS[layer]}.{param}"


def convert_proprio_projector_key(key: str) -> str:
    """Maps a key of the released `ProprioProjector` state dict to `OpenVLAOFT`."""
    return "proprio_projector." + key.removeprefix("module.")


def pruned_prefixes(model: OpenVLAOFT) -> tuple[str, ...]:
    """Prefixes of released vision weights that `OpenVLAOFT` removed because they are unused."""
    prefixes = []
    for name in ("dinov2", "siglip"):
        vit = getattr(model.vision, name).vit
        prefixes += [
            f"vision.{name}.vit.{part}." for part in (f"blocks.{len(vit.blocks)}", "norm", "attn_pool")
        ]
    return tuple(prefixes)


def load_released_weights(
    model: OpenVLAOFT,
    vla: dict[str, Tensor],
    action_head: dict[str, Tensor],
    proprio_projector: dict[str, Tensor],
) -> None:
    """Loads released weights into `model`, requiring every parameter to be covered.

    Weights that this port intentionally removed (the language-model head and the
    unused vision layers) are dropped; any other mismatch raises an error.
    """
    converted = {new: value for key, value in vla.items() if (new := convert_vla_key(key)) is not None}
    converted |= {convert_action_head_key(k): v for k, v in action_head.items()}
    converted |= {convert_proprio_projector_key(k): v for k, v in proprio_projector.items()}

    expected = model.state_dict().keys()
    unexpected = [k for k in converted.keys() - expected if not k.startswith(pruned_prefixes(model))]
    if unexpected:
        raise KeyError(f"Checkpoint keys without a destination: {sorted(unexpected)}")
    model.load_state_dict({k: v for k, v in converted.items() if k in expected}, strict=True)


def convert_dataset_statistics(path: Path) -> tuple[dict[str, dict[str, Tensor]], list[bool]]:
    """Reads `dataset_statistics.json` into LeRobot statistics and the action mask."""
    (statistics,) = json.loads(path.read_text()).values()
    stats = {
        key: {name: torch.tensor(values) for name, values in statistics[source].items() if name != "mask"}
        for key, source in ((ACTION, "action"), (OBS_STATE, "proprio"))
    }
    return stats, [bool(m) for m in statistics["action"]["mask"]]


def libero_config(stats: dict[str, dict[str, Tensor]], action_norm_mask: list[bool]) -> OpenVLAOFTConfig:
    """Configuration matching the released LIBERO checkpoints and LeRobot's LIBERO environment."""
    image = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
    return OpenVLAOFTConfig(
        input_features={
            **dict.fromkeys(LIBERO_IMAGE_KEYS, image),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=tuple(stats[OBS_STATE]["q01"].shape)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=tuple(stats[ACTION]["q01"].shape))
        },
        action_norm_mask=action_norm_mask,
    )


def convert(repo_id: str, output_dir: Path, revision: str | None = None) -> None:
    checkpoint = Path(snapshot_download(repo_id, revision=revision, allow_patterns=CHECKPOINT_FILES))
    stats, action_norm_mask = convert_dataset_statistics(checkpoint / "dataset_statistics.json")
    config = libero_config(stats, action_norm_mask)

    policy = OpenVLAOFTPolicy(config)
    vla: dict[str, Tensor] = {}
    for shard in sorted(checkpoint.glob("model-*.safetensors")):
        vla |= load_file(shard)
    (action_head,) = checkpoint.glob("action_head--*.pt")
    (proprio_projector,) = checkpoint.glob("proprio_projector--*.pt")
    load_released_weights(
        policy.model,
        vla,
        torch.load(action_head, weights_only=True),
        torch.load(proprio_projector, weights_only=True),
    )

    preprocessor, postprocessor = make_openvla_oft_pre_post_processors(config, stats)
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-id", required=True, help="Released checkpoint on the Hugging Face Hub.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--revision", help="Optional commit, branch, or tag of the checkpoint repository.")
    args = parser.parse_args()
    convert(args.repo_id, args.output_dir, args.revision)


if __name__ == "__main__":
    main()
