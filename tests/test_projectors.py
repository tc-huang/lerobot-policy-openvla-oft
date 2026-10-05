import torch

from lerobot_policy_openvla_oft.projectors import ProprioProjector, VisionProjector


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def test_vision_projector_maps_each_patch_to_llm_dim():
    projector = VisionProjector(vision_dim=6, llm_dim=8)

    assert projector(torch.randn(2, 5, 6)).shape == (2, 5, 8)


def test_proprio_projector_produces_one_token():
    projector = ProprioProjector(proprio_dim=3, llm_dim=8)

    assert projector(torch.randn(2, 3)).shape == (2, 1, 8)


def test_proprio_projector_size_matches_paper():
    with torch.device("meta"):
        projector = ProprioProjector(proprio_dim=8, llm_dim=4096)

    assert round(count_parameters(projector) / 1e6) == 17
