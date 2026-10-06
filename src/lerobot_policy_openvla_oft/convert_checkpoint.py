"""Converts a released OpenVLA or OpenVLA-OFT checkpoint into a LeRobot policy directory.

Usage:
    python -m lerobot_policy_openvla_oft.convert_checkpoint \\
        --repo-id moojink/openvla-7b-oft-finetuned-libero-spatial \\
        --output-dir outputs/checkpoints/libero-spatial

    python -m lerobot_policy_openvla_oft.convert_checkpoint --base \\
        --repo-id openvla/openvla-7b \\
        --output-dir outputs/checkpoints/openvla-7b
"""

import argparse
import json
import re
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from safetensors.torch import load_file, save_file
from torch import Tensor

from .configuration_openvla_oft import OpenVLAOFTConfig
from .language_model import BidirectionalLlama, openvla_llama_config
from .model import OpenVLAOFT
from .modeling_openvla_oft import OpenVLAOFTPolicy
from .processor_openvla_oft import OpenVLALiberoGripperProcessorStep, make_openvla_oft_pre_post_processors
from .vision_backbone import FusedVisionBackbone

BASE_FILES = ["model.safetensors.index.json", "model-*.safetensors"]
CHECKPOINT_FILES = [*BASE_FILES, "action_head--*.pt", "proprio_projector--*.pt", "dataset_statistics.json"]
NEW_MODULES = ("action_head.", "proprio_projector.")

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


def convert_released_weights(
    model: OpenVLAOFT,
    vla: dict[str, Tensor],
    action_head: dict[str, Tensor] | None = None,
    proprio_projector: dict[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    """Maps released weights onto `model`'s parameter names, requiring every one to be covered.

    Weights that this port intentionally removed (the language-model head and the
    unused vision layers) are dropped; any other mismatch raises an error. Without
    `action_head` and `proprio_projector`, as for the base OpenVLA model, the result
    covers only the vision backbone, the vision projector, and the language model.
    """
    converted = {new: value for key, value in vla.items() if (new := convert_vla_key(key)) is not None}
    converted |= {convert_action_head_key(k): v for k, v in (action_head or {}).items()}
    converted |= {convert_proprio_projector_key(k): v for k, v in (proprio_projector or {}).items()}

    expected = set(model.state_dict())
    if action_head is None and proprio_projector is None:
        expected = {k for k in expected if not k.startswith(NEW_MODULES)}
    unexpected = [k for k in converted.keys() - expected if not k.startswith(pruned_prefixes(model))]
    if unexpected:
        raise KeyError(f"Checkpoint keys without a destination: {sorted(unexpected)}")
    missing = expected - converted.keys()
    if missing:
        raise KeyError(f"Parameters missing from the checkpoint: {sorted(missing)}")
    return {k: converted[k] for k in expected}


def load_released_weights(
    model: OpenVLAOFT,
    vla: dict[str, Tensor],
    action_head: dict[str, Tensor],
    proprio_projector: dict[str, Tensor],
) -> None:
    """Loads a released OpenVLA-OFT checkpoint into `model`."""
    model.load_state_dict(convert_released_weights(model, vla, action_head, proprio_projector), strict=True)


def save_base_policy(output_dir: Path, config: OpenVLAOFTConfig, weights: dict[str, Tensor]) -> None:
    """Writes a policy directory with only pretrained OpenVLA weights.

    The configuration has no input or output features, so `lerobot-train` fills them
    in from the dataset, and LeRobot's non-strict loading leaves the action head and
    proprio projector, which the base model does not have, at their initialization.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file({f"model.{k}": v.contiguous() for k, v in weights.items()}, output_dir / "model.safetensors")
    config.save_pretrained(output_dir)
    preprocessor, postprocessor = make_openvla_oft_pre_post_processors(config)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)


def convert_dataset_statistics(path: Path) -> dict[str, dict[str, Tensor]]:
    """Reads `dataset_statistics.json` into LeRobot statistics for state and action.

    The original leaves action dimensions whose `mask` entry is False unnormalized.
    LeRobot's `QUANTILES` mode maps [q01, q99] to [-1, 1], which is the identity for
    q01 = -1 and q99 = 1, so masked dimensions get those quantiles.
    """
    (statistics,) = json.loads(path.read_text()).values()
    stats = {
        key: {name: torch.tensor(values) for name, values in statistics[source].items() if name != "mask"}
        for key, source in ((ACTION, "action"), (OBS_STATE, "proprio"))
    }
    unnormalized = ~torch.tensor(statistics["action"]["mask"])
    stats[ACTION]["q01"] = torch.where(unnormalized, -1.0, stats[ACTION]["q01"])
    stats[ACTION]["q99"] = torch.where(unnormalized, 1.0, stats[ACTION]["q99"])
    return stats


def libero_config(stats: dict[str, dict[str, Tensor]]) -> OpenVLAOFTConfig:
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
    )


def libero_processors(
    config: OpenVLAOFTConfig, stats: dict[str, dict[str, Tensor]]
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Policy processors plus the gripper conversion that LIBERO evaluation needs."""
    preprocessor, postprocessor = make_openvla_oft_pre_post_processors(config, stats)
    postprocessor.steps = [*postprocessor.steps, OpenVLALiberoGripperProcessorStep()]
    return preprocessor, postprocessor


def load_vla_shards(checkpoint: Path) -> dict[str, Tensor]:
    vla: dict[str, Tensor] = {}
    for shard in sorted(checkpoint.glob("model-*.safetensors")):
        vla |= load_file(shard)
    return vla


def convert(repo_id: str, output_dir: Path, revision: str | None = None) -> None:
    """Converts a released OpenVLA-OFT LIBERO checkpoint."""
    checkpoint = Path(snapshot_download(repo_id, revision=revision, allow_patterns=CHECKPOINT_FILES))
    stats = convert_dataset_statistics(checkpoint / "dataset_statistics.json")
    config = libero_config(stats)

    policy = OpenVLAOFTPolicy(config)
    (action_head,) = checkpoint.glob("action_head--*.pt")
    (proprio_projector,) = checkpoint.glob("proprio_projector--*.pt")
    load_released_weights(
        policy.model,
        load_vla_shards(checkpoint),
        torch.load(action_head, map_location="cpu", weights_only=True),
        torch.load(proprio_projector, map_location="cpu", weights_only=True),
    )

    preprocessor, postprocessor = libero_processors(config, stats)
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)


def convert_base(repo_id: str, output_dir: Path, revision: str | None = None) -> None:
    """Converts the base OpenVLA model into a starting point for fine-tuning."""
    checkpoint = Path(snapshot_download(repo_id, revision=revision, allow_patterns=BASE_FILES))
    config = OpenVLAOFTConfig()
    with torch.device("meta"):
        model = OpenVLAOFT(
            FusedVisionBackbone(config.image_size),
            BidirectionalLlama(openvla_llama_config()),
            config.chunk_size,
            action_dim=1,
            proprio_dim=None,
        )
    save_base_policy(output_dir, config, convert_released_weights(model, load_vla_shards(checkpoint)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-id", required=True, help="Released checkpoint on the Hugging Face Hub.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--revision", help="Optional commit, branch, or tag of the checkpoint repository.")
    parser.add_argument(
        "--base", action="store_true", help="Convert the base OpenVLA model (no action head) for fine-tuning."
    )
    args = parser.parse_args()
    (convert_base if args.base else convert)(args.repo_id, args.output_dir, args.revision)


if __name__ == "__main__":
    main()
