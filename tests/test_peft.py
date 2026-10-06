import dataclasses
import re

import pytest
import torch
from lerobot.configs.default import PeftConfig
from peft import PeftModel
from released_format import released_checkpoint
from tiny_policy import build_tiny_model, make_batch, make_config

from lerobot_policy_openvla_oft import OpenVLAOFTConfig, OpenVLAOFTPolicy, modeling_openvla_oft
from lerobot_policy_openvla_oft.convert_checkpoint import convert_released_weights, save_base_policy
from lerobot_policy_openvla_oft.language_model import BidirectionalLlama, openvla_llama_config
from lerobot_policy_openvla_oft.model import OpenVLAOFT
from lerobot_policy_openvla_oft.modeling_openvla_oft import LORA_TARGET_MODULES
from lerobot_policy_openvla_oft.vision_backbone import FusedVisionBackbone

VLA_MODULES = ("model.vision.", "model.vision_projector.", "model.llm.")


@pytest.fixture(autouse=True)
def tiny_build_model(monkeypatch):
    monkeypatch.setattr(modeling_openvla_oft, "build_model", build_tiny_model)


def wrap_like_lerobot_train(policy, **cli):
    """Mirrors `lerobot_train.py`, which passes every `--peft.*` field, defaults included."""
    return policy.wrap_with_peft(peft_cli_overrides=dataclasses.asdict(PeftConfig(**cli)))


def test_lora_targets_are_exactly_the_linear_layers_of_the_full_size_vla():
    with torch.device("meta"):
        model = OpenVLAOFT(FusedVisionBackbone(224), BidirectionalLlama(openvla_llama_config()), 8, 7, 8)
    names = {f"model.{name}": module for name, module in model.named_modules()}

    targeted = {name for name in names if re.fullmatch(LORA_TARGET_MODULES, name)}
    linear = {n for n, m in names.items() if isinstance(m, torch.nn.Linear) and n.startswith(VLA_MODULES)}

    assert targeted == linear


def test_trains_lora_and_new_modules_only():
    policy = OpenVLAOFTPolicy(make_config(pretrained_path="base"))

    peft_policy = wrap_like_lerobot_train(policy, r=4)

    trainable = {n for n, p in peft_policy.named_parameters() if p.requires_grad}
    assert trainable
    assert all("lora_" in n or ".modules_to_save." in n for n in trainable)
    assert any("action_head.modules_to_save" in n for n in trainable)
    assert any("proprio_projector.modules_to_save" in n for n in trainable)
    lora_layers = {n.split(".lora_A")[0] for n in trainable if ".lora_A." in n}
    linear_layers = [
        m for n, m in policy.named_modules() if isinstance(m, torch.nn.Linear) and ".base_layer" in n
    ]
    assert len(lora_layers) == len(linear_layers)


def test_uses_original_lora_hyperparameters():
    policy = OpenVLAOFTPolicy(make_config(pretrained_path="base"))

    config = wrap_like_lerobot_train(policy, r=32).peft_config["default"]

    assert (config.r, config.lora_alpha, config.lora_dropout) == (32, 16, 0.0)
    assert config.init_lora_weights == "gaussian"


def test_adapter_round_trip(tmp_path):
    peft_policy = wrap_like_lerobot_train(OpenVLAOFTPolicy(make_config(pretrained_path="base")), r=4)
    loss, _ = peft_policy(make_batch())
    loss.backward()
    with torch.no_grad():
        for param in peft_policy.parameters():
            if param.requires_grad:
                param.add_(torch.randn_like(param) * 0.1)
    peft_policy.eval()
    peft_policy.save_pretrained(tmp_path)
    batch = make_batch()

    reloaded = PeftModel.from_pretrained(OpenVLAOFTPolicy(make_config()), tmp_path).eval()

    torch.testing.assert_close(reloaded.predict_action_chunk(batch), peft_policy.predict_action_chunk(batch))


def test_base_policy_loads_vla_weights_and_initializes_new_modules(tmp_path):
    config = make_config()
    source = build_tiny_model(config)
    vla, _, _ = released_checkpoint(source)
    torch.manual_seed(1)
    target = build_tiny_model(config)

    save_base_policy(
        tmp_path, OpenVLAOFTConfig(device="cpu", dtype="float32"), convert_released_weights(target, vla)
    )
    policy = OpenVLAOFTPolicy.from_pretrained(tmp_path, config=config)

    loaded = policy.model.state_dict()
    for key, value in source.state_dict().items():
        if key.startswith(("action_head.", "proprio_projector.")):
            continue
        torch.testing.assert_close(loaded[key], value, rtol=0, atol=0)
    assert OpenVLAOFTConfig.from_pretrained(tmp_path).input_features == {}
    assert not any(t.is_meta for t in [*policy.parameters(), *policy.buffers()])
    for module in (policy.model.action_head, policy.model.proprio_projector):
        weights = torch.cat([p.flatten() for p in module.parameters()])
        assert torch.isfinite(weights).all() and weights.std() > 0
