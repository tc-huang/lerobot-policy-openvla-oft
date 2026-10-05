from functools import partial

import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from tiny_models import TINY_IMAGE_SIZE, TINY_LLAMA, TINY_VIT

from lerobot_policy_openvla_oft import (
    OpenVLAOFTConfig,
    OpenVLAOFTPolicy,
    make_openvla_oft_pre_post_processors,
    modeling_openvla_oft,
)
from lerobot_policy_openvla_oft.language_model import BidirectionalLlama, openvla_llama_config
from lerobot_policy_openvla_oft.model import OpenVLAOFT
from lerobot_policy_openvla_oft.vision_backbone import FusedVisionBackbone

CHUNK_SIZE, ACTION_DIM, STATE_DIM, BATCH_SIZE = 4, 3, 5, 2
IMAGE_KEYS = (f"{OBS_IMAGES}.image", f"{OBS_IMAGES}.wrist_image")


def build_tiny_model(config, **llama_overrides):
    torch.manual_seed(0)
    state = config.robot_state_feature
    return OpenVLAOFT(
        vision=FusedVisionBackbone(TINY_IMAGE_SIZE, **TINY_VIT),
        llm=BidirectionalLlama(openvla_llama_config(**(TINY_LLAMA | llama_overrides))),
        chunk_size=config.chunk_size,
        action_dim=config.action_feature.shape[0],
        proprio_dim=state.shape[0] if state is not None else None,
    )


@pytest.fixture(autouse=True)
def tiny_build_model(monkeypatch):
    monkeypatch.setattr(modeling_openvla_oft, "build_model", build_tiny_model)


def make_config(with_state=True, **overrides):
    image = PolicyFeature(type=FeatureType.VISUAL, shape=(3, TINY_IMAGE_SIZE, TINY_IMAGE_SIZE))
    input_features = dict.fromkeys(IMAGE_KEYS, image)
    if with_state:
        input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))
    return OpenVLAOFTConfig(
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        image_size=TINY_IMAGE_SIZE,
        device="cpu",
        **{"dtype": "float32", **overrides},
    )


def make_batch():
    prompt_mask = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
    batch = {key: torch.rand(BATCH_SIZE, 3, TINY_IMAGE_SIZE, TINY_IMAGE_SIZE) for key in IMAGE_KEYS}
    batch[OBS_STATE] = torch.randn(BATCH_SIZE, STATE_DIM)
    batch[OBS_LANGUAGE_TOKENS] = torch.randint(3, 64, prompt_mask.shape).masked_fill(~prompt_mask, 0)
    batch[OBS_LANGUAGE_TOKENS][:, 0] = 1
    batch[OBS_LANGUAGE_ATTENTION_MASK] = prompt_mask.long()
    batch[ACTION] = torch.randn(BATCH_SIZE, CHUNK_SIZE, ACTION_DIM)
    batch[f"{ACTION}_is_pad"] = torch.zeros(BATCH_SIZE, CHUNK_SIZE, dtype=torch.bool)
    return batch


def test_resolved_by_lerobot_factory():
    assert get_policy_class("openvla_oft") is OpenVLAOFTPolicy


def test_forward_returns_differentiable_l1_loss():
    policy = OpenVLAOFTPolicy(make_config()).eval()
    batch = make_batch()

    loss, info = policy(batch)
    loss.backward()

    expected = (policy.predict_action_chunk(batch) - batch[ACTION]).abs().mean()
    torch.testing.assert_close(loss.detach(), expected)
    assert info == {"l1_loss": pytest.approx(loss.item())}
    assert all(p.grad is not None for p in policy.get_optim_params())


@pytest.mark.parametrize("mask_padded_actions", [False, True])
def test_padded_actions_count_only_when_unmasked(mask_padded_actions):
    policy = OpenVLAOFTPolicy(make_config(mask_padded_actions=mask_padded_actions)).eval()
    batch = make_batch()
    batch[f"{ACTION}_is_pad"][:, -1] = True
    loss, _ = policy(batch)

    batch[ACTION][:, -1] += 100.0
    loss_after_change, _ = policy(batch)

    assert torch.isclose(loss, loss_after_change).item() is mask_padded_actions


def test_select_action_replays_chunk_before_predicting_again(monkeypatch):
    policy = OpenVLAOFTPolicy(make_config()).eval()
    chunk = torch.randn(BATCH_SIZE, CHUNK_SIZE, ACTION_DIM)
    calls = []
    monkeypatch.setattr(policy, "predict_action_chunk", lambda batch: calls.append(1) or chunk)

    actions = [policy.select_action(make_batch()) for _ in range(CHUNK_SIZE + 1)]

    assert len(calls) == 2
    for step in range(CHUNK_SIZE):
        torch.testing.assert_close(actions[step], chunk[:, step])

    policy.reset()
    policy.select_action(make_batch())
    assert len(calls) == 3


def test_default_precision_keeps_proprio_projector_in_float32():
    policy = OpenVLAOFTPolicy(make_config(dtype="bfloat16"))

    assert {p.dtype for p in policy.model.proprio_projector.parameters()} == {torch.float32}
    assert {p.dtype for p in policy.model.llm.parameters()} == {torch.bfloat16}
    assert policy.predict_action_chunk(make_batch()).shape == (BATCH_SIZE, CHUNK_SIZE, ACTION_DIM)


def test_proprio_projector_can_follow_model_dtype():
    policy = OpenVLAOFTPolicy(make_config(dtype="bfloat16", proprio_projector_fp32=False))

    assert {p.dtype for p in policy.parameters()} == {torch.bfloat16}


def test_runs_without_state_feature():
    policy = OpenVLAOFTPolicy(make_config(with_state=False))
    batch = make_batch()
    del batch[OBS_STATE]

    assert policy.model.proprio_projector is None
    assert policy.predict_action_chunk(batch).shape == (BATCH_SIZE, CHUNK_SIZE, ACTION_DIM)


def test_save_and_load_round_trip(tmp_path):
    policy = OpenVLAOFTPolicy(make_config(dtype="bfloat16", push_to_hub=False)).eval()
    batch = make_batch()
    policy.save_pretrained(tmp_path)

    loaded = OpenVLAOFTPolicy.from_pretrained(tmp_path)

    torch.testing.assert_close(loaded.predict_action_chunk(batch), policy.predict_action_chunk(batch))
    assert {p.dtype for p in loaded.model.proprio_projector.parameters()} == {torch.float32}


def test_accepts_preprocessor_output(monkeypatch):
    monkeypatch.setattr(modeling_openvla_oft, "build_model", partial(build_tiny_model, vocab_size=32064))
    config = make_config()
    stats = {"q01": torch.tensor(-1.0), "q99": torch.tensor(1.0)}
    preprocessor, postprocessor = make_openvla_oft_pre_post_processors(
        config, {OBS_STATE: stats, ACTION: stats}
    )
    policy = OpenVLAOFTPolicy(config).eval()
    observation = {key: torch.rand(3, TINY_IMAGE_SIZE, TINY_IMAGE_SIZE) for key in IMAGE_KEYS}
    observation |= {OBS_STATE: torch.zeros(STATE_DIM), "task": "Open the drawer"}

    action = postprocessor(policy.select_action(preprocessor(observation)))

    assert action.shape == (1, ACTION_DIM)


def test_crops_randomly_only_in_training():
    policy = OpenVLAOFTPolicy(make_config())
    batch = make_batch()

    assert not torch.equal(policy._predict(batch), policy._predict(batch))
    policy.eval()
    torch.testing.assert_close(policy._predict(batch), policy._predict(batch))
