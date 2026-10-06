import json

import pytest
import torch
from lerobot.utils.constants import ACTION, OBS_STATE
from released_format import released_checkpoint

from lerobot_policy_openvla_oft.convert_checkpoint import (
    convert_action_head_key,
    convert_dataset_statistics,
    convert_vla_key,
    libero_config,
    libero_processors,
    load_released_weights,
)
from lerobot_policy_openvla_oft.model import OpenVLAOFT
from lerobot_policy_openvla_oft.processor_openvla_oft import make_openvla_oft_pre_post_processors


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

    with pytest.raises(KeyError, match="missing from the checkpoint.*vision_projector.fc1.weight"):
        load_released_weights(make_model(1), vla, action_head, proprio)


def test_dataset_statistics_become_lerobot_stats(tmp_path):
    path = tmp_path / "dataset_statistics.json"
    action = {"q01": [-0.5, 0.0], "q99": [0.5, 1.0], "mask": [True, False]}
    path.write_text(
        json.dumps({"libero_spatial_no_noops": {"action": action, "proprio": {"q01": [0.0], "q99": [2.0]}}})
    )

    stats = convert_dataset_statistics(path)
    config = libero_config(stats)

    torch.testing.assert_close(stats[ACTION]["q01"], torch.tensor([-0.5, -1.0]))
    torch.testing.assert_close(stats[ACTION]["q99"], torch.tensor([0.5, 1.0]))
    torch.testing.assert_close(stats[OBS_STATE]["q99"], torch.tensor([2.0]))
    assert config.robot_state_feature.shape == (1,)
    assert config.action_feature.shape == (2,)
    assert list(config.image_features) == ["observation.images.image", "observation.images.image2"]


def test_masked_action_dims_pass_through_unchanged(tmp_path):
    path = tmp_path / "dataset_statistics.json"
    action = {"q01": [-0.5] * 7, "q99": [0.5] * 7, "mask": [True] * 6 + [False]}
    path.write_text(
        json.dumps({"suite": {"action": action, "proprio": {"q01": [0.0] * 8, "q99": [1.0] * 8}}})
    )
    stats = convert_dataset_statistics(path)
    config = libero_config(stats)
    config.device = "cpu"
    preprocessor, postprocessor = make_openvla_oft_pre_post_processors(config, stats)
    observation = {key: torch.rand(3, 224, 224) for key in config.image_features}
    gripper = torch.tensor([0.0, 0.3, 1.0])

    actions = torch.zeros(3, 7)
    actions[:, -1] = gripper
    normalized = preprocessor(observation | {OBS_STATE: torch.zeros(8), ACTION: actions, "task": "x"})[ACTION]

    torch.testing.assert_close(normalized[:, -1], gripper)
    torch.testing.assert_close(postprocessor(normalized)[:, -1], gripper)


def test_libero_postprocessor_converts_gripper_last():
    stats = {
        key: {"q01": torch.zeros(dim), "q99": torch.ones(dim)} for key, dim in ((ACTION, 7), (OBS_STATE, 8))
    }
    config = libero_config(stats)

    _, postprocessor = libero_processors(config, stats)

    action = postprocessor(torch.tensor([[0.0] * 6 + [0.9]]))
    assert action[0, -1].item() == -1.0
