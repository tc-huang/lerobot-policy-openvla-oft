import pytest
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from lerobot_policy_openvla_oft import OpenVLAOFTConfig


def test_registered_as_policy_type():
    assert PreTrainedConfig.get_choice_class("openvla_oft") is OpenVLAOFTConfig


def test_action_delta_indices_cover_chunk():
    config = OpenVLAOFTConfig(chunk_size=8)
    assert config.action_delta_indices == list(range(8))
    assert config.observation_delta_indices is None


@pytest.mark.parametrize("n_action_steps", [0, 9])
def test_rejects_n_action_steps_outside_chunk(n_action_steps):
    with pytest.raises(ValueError, match="n_action_steps"):
        OpenVLAOFTConfig(chunk_size=8, n_action_steps=n_action_steps)


def test_validate_features_requires_image_and_action():
    image = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
    state = PolicyFeature(type=FeatureType.STATE, shape=(8,))
    action = PolicyFeature(type=FeatureType.ACTION, shape=(7,))

    OpenVLAOFTConfig(
        input_features={f"{OBS_IMAGES}.image": image, OBS_STATE: state},
        output_features={ACTION: action},
    ).validate_features()

    with pytest.raises(ValueError, match="image"):
        OpenVLAOFTConfig(
            input_features={OBS_STATE: state}, output_features={ACTION: action}
        ).validate_features()

    with pytest.raises(ValueError, match="action"):
        OpenVLAOFTConfig(input_features={f"{OBS_IMAGES}.image": image}).validate_features()


def test_optimizer_preset_disables_grad_clipping_by_default():
    optimizer = OpenVLAOFTConfig().get_optimizer_preset()
    assert optimizer.lr == 5e-4
    assert optimizer.grad_clip_norm == 0.0


def test_save_and_load_round_trip(tmp_path):
    config = OpenVLAOFTConfig(chunk_size=4, n_action_steps=2, push_to_hub=False)
    config.save_pretrained(tmp_path)
    assert PreTrainedConfig.from_pretrained(tmp_path) == config


def test_scheduler_preset_decays_learning_rate_once():
    config = OpenVLAOFTConfig(scheduler_decay_steps=3)
    model = torch.nn.Linear(1, 1)
    optimizer = config.get_optimizer_preset().build(model.parameters())
    scheduler = config.get_scheduler_preset().build(optimizer, num_training_steps=10)

    lrs = []
    for _ in range(6):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert lrs == pytest.approx([5e-4] * 3 + [5e-5] * 3)


def test_rejects_unsupported_dtype():
    with pytest.raises(ValueError, match="dtype"):
        OpenVLAOFTConfig(dtype="float16")
