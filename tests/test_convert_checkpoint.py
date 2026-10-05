import json

import pytest
import torch
from lerobot.utils.constants import ACTION, OBS_STATE

from lerobot_policy_openvla_oft.convert_checkpoint import (
    convert_action_head_key,
    convert_dataset_statistics,
    convert_vla_key,
    libero_config,
    libero_processors,
    load_released_weights,
)
from lerobot_policy_openvla_oft.model import OpenVLAOFT

RELEASED_ACTION_HEAD_LAYERS = {
    "input_norm": "layer_norm1",
    "input_proj": "fc1",
    "output_norm": "layer_norm2",
    "output_proj": "fc2",
}


def released_key(key):
    """Inverse of the conversion, written independently from the released module names."""
    if key.startswith("action_head.blocks."):
        _, _, index, layer, param = key.split(".")
        return f"module.model.mlp_resnet_blocks.{index}.ffn.{0 if layer == 'norm' else 1}.{param}"
    if key.startswith("action_head."):
        _, layer, param = key.split(".")
        return f"module.model.{RELEASED_ACTION_HEAD_LAYERS[layer]}.{param}"
    if key.startswith("proprio_projector."):
        return "module." + key.removeprefix("proprio_projector.")
    key = key.replace("vision.dinov2.vit.", "vision_backbone.featurizer.")
    key = key.replace("vision.siglip.vit.", "vision_backbone.fused_featurizer.")
    key = key.replace("vision_projector.", "projector.").replace("llm.model.", "language_model.model.")
    return key.replace(".ls1.gamma", ".ls1.scale_factor").replace(".ls2.gamma", ".ls2.scale_factor")


def released_checkpoint(model):
    vla, action_head, proprio = {}, {}, {}
    for key, value in model.state_dict().items():
        target = (
            action_head if key.startswith("action_head.") else proprio if key.startswith("proprio") else vla
        )
        target[released_key(key)] = value.clone()
    pruned_block = len(model.vision.dinov2.vit.blocks)
    vla |= {
        "language_model.lm_head.weight": torch.zeros(1),
        f"vision_backbone.featurizer.blocks.{pruned_block}.attn.qkv.weight": torch.zeros(1),
        "vision_backbone.featurizer.norm.weight": torch.zeros(1),
        "vision_backbone.fused_featurizer.attn_pool.latent": torch.zeros(1),
    }
    return vla, action_head, proprio


@pytest.fixture
def make_model(tiny_vision, tiny_llm):
    def make(seed):
        torch.manual_seed(seed)
        return OpenVLAOFT(tiny_vision, tiny_llm, chunk_size=2, action_dim=3, proprio_dim=4)

    return make


def test_vla_keys():
    assert convert_vla_key("vision_backbone.featurizer.blocks.0.ls1.scale_factor") == (
        "vision.dinov2.vit.blocks.0.ls1.gamma"
    )
    assert convert_vla_key("vision_backbone.fused_featurizer.pos_embed") == "vision.siglip.vit.pos_embed"
    assert convert_vla_key("projector.fc3.bias") == "vision_projector.fc3.bias"
    assert convert_vla_key("language_model.model.norm.weight") == "llm.model.norm.weight"
    assert convert_vla_key("language_model.lm_head.weight") is None
    with pytest.raises(KeyError):
        convert_vla_key("unknown.weight")


def test_action_head_keys():
    assert convert_action_head_key("module.model.fc1.weight") == "action_head.input_proj.weight"
    assert (
        convert_action_head_key("module.model.mlp_resnet_blocks.1.ffn.0.bias")
        == "action_head.blocks.1.norm.bias"
    )
    assert convert_action_head_key("module.model.mlp_resnet_blocks.1.ffn.1.weight") == (
        "action_head.blocks.1.linear.weight"
    )


def test_loads_every_released_weight(make_model):
    source, target = make_model(0), make_model(1)

    load_released_weights(target, *released_checkpoint(source))

    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)


def test_rejects_unknown_weights(make_model):
    vla, action_head, proprio = released_checkpoint(make_model(0))
    vla["projector.fc4.weight"] = torch.zeros(1)

    with pytest.raises(KeyError, match="vision_projector.fc4.weight"):
        load_released_weights(make_model(1), vla, action_head, proprio)


def test_rejects_missing_weights(make_model):
    vla, action_head, proprio = released_checkpoint(make_model(0))
    del vla["projector.fc1.weight"]

    with pytest.raises(RuntimeError, match="vision_projector.fc1.weight"):
        load_released_weights(make_model(1), vla, action_head, proprio)


def test_dataset_statistics_become_lerobot_stats(tmp_path):
    path = tmp_path / "dataset_statistics.json"
    action = {"q01": [0.0, 0.0], "q99": [1.0, 1.0], "mask": [True, False]}
    path.write_text(
        json.dumps({"libero_spatial_no_noops": {"action": action, "proprio": {"q01": [0.0], "q99": [2.0]}}})
    )

    stats, mask = convert_dataset_statistics(path)
    config = libero_config(stats, mask)

    assert mask == [True, False]
    assert set(stats[ACTION]) == {"q01", "q99"}
    torch.testing.assert_close(stats[OBS_STATE]["q99"], torch.tensor([2.0]))
    assert config.robot_state_feature.shape == (1,)
    assert config.action_feature.shape == (2,)
    assert list(config.image_features) == ["observation.images.image", "observation.images.image2"]


def test_libero_postprocessor_converts_gripper_last():
    stats = {
        key: {"q01": torch.zeros(dim), "q99": torch.ones(dim)} for key, dim in ((ACTION, 7), (OBS_STATE, 8))
    }
    config = libero_config(stats, [True] * 6 + [False])

    _, postprocessor = libero_processors(config, stats)

    action = postprocessor(torch.tensor([[0.0] * 6 + [0.9]]))
    assert action[0, -1].item() == -1.0
