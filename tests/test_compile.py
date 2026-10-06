import dataclasses

import pytest
import torch
from lerobot.configs.default import PeftConfig
from tiny_policy import build_tiny_model, make_batch, make_config

from lerobot_policy_openvla_oft import OpenVLAOFTPolicy, modeling_openvla_oft


@pytest.fixture(autouse=True)
def tiny_build_model(monkeypatch):
    monkeypatch.setattr(modeling_openvla_oft, "build_model", build_tiny_model)
    torch._dynamo.reset()


def test_compiled_policy_matches_eager_and_keeps_parameter_names():
    eager = OpenVLAOFTPolicy(make_config()).eval()
    compiled = OpenVLAOFTPolicy(make_config(compile_model=True)).eval()
    batch = make_batch()

    torch.testing.assert_close(compiled.predict_action_chunk(batch), eager.predict_action_chunk(batch))
    assert compiled.state_dict().keys() == eager.state_dict().keys()
    assert compiled.model._compiled_call_impl is not None
    assert eager.model._compiled_call_impl is None


def test_compiled_policy_trains_with_lora():
    policy = OpenVLAOFTPolicy(make_config(compile_model=True, pretrained_path="base"))
    peft_policy = policy.wrap_with_peft(peft_cli_overrides=dataclasses.asdict(PeftConfig(r=4)))

    loss, _ = peft_policy(make_batch())
    loss.backward()

    trainable = [p for p in peft_policy.parameters() if p.requires_grad]
    assert all(p.grad is not None for p in trainable)
