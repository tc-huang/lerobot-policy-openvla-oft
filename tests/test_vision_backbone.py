import importlib.util
from pathlib import Path

import pytest
import timm
import torch
from tiny_models import TINY_IMAGE_SIZE, TINY_VIT
from torch import nn

from lerobot_policy_openvla_oft.vision_backbone import (
    DINOV2_MEAN,
    DINOV2_MODEL_ID,
    DINOV2_STD,
    FusedVisionBackbone,
    PatchFeaturizer,
)

FILM_DIM = 8
ORIGINAL_FILM = Path(__file__).parents[1] / "third_party/openvla-oft/prismatic/models/film_vit_wrapper.py"


def normalize(images):
    mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in (DINOV2_MEAN, DINOV2_STD))
    return (images - mean) / std


def test_output_shape_concatenates_channels_and_images(tiny_vision):
    images = torch.rand(2, 3, 3, TINY_IMAGE_SIZE, TINY_IMAGE_SIZE)

    features = tiny_vision(images)

    assert tiny_vision.num_patches == (TINY_IMAGE_SIZE // 14) ** 2
    assert tiny_vision.embed_dim == 2 * TINY_VIT["embed_dim"]
    assert features.shape == (2, 3 * tiny_vision.num_patches, tiny_vision.embed_dim)


def test_matches_second_to_last_block_of_unpruned_vit():
    torch.manual_seed(0)
    reference = timm.create_model(
        DINOV2_MODEL_ID, pretrained=False, num_classes=0, img_size=TINY_IMAGE_SIZE, **TINY_VIT
    ).eval()
    torch.manual_seed(0)
    featurizer = PatchFeaturizer(DINOV2_MODEL_ID, TINY_IMAGE_SIZE, DINOV2_MEAN, DINOV2_STD, **TINY_VIT).eval()
    images = torch.rand(2, 3, TINY_IMAGE_SIZE, TINY_IMAGE_SIZE)

    expected = reference.get_intermediate_layers(normalize(images), n={TINY_VIT["depth"] - 2})[0]

    torch.testing.assert_close(featurizer(images), expected)
    assert len(featurizer.vit.blocks) == TINY_VIT["depth"] - 1


def test_each_image_is_encoded_independently(tiny_vision):
    tiny_vision.eval()
    images = torch.rand(1, 2, 3, TINY_IMAGE_SIZE, TINY_IMAGE_SIZE)

    features = tiny_vision(images)
    per_image = [tiny_vision(images[:, i : i + 1]) for i in range(2)]

    torch.testing.assert_close(features, torch.cat(per_image, dim=1))


def test_every_parameter_receives_gradient(tiny_vision):
    tiny_vision(torch.rand(1, 1, 3, TINY_IMAGE_SIZE, TINY_IMAGE_SIZE)).sum().backward()

    assert all(p.grad is not None for p in tiny_vision.parameters())


def test_film_matches_original_implementation():
    if not ORIGINAL_FILM.exists():
        pytest.skip("The openvla-oft submodule is not checked out.")
    spec = importlib.util.spec_from_file_location("original_film_vit_wrapper", ORIGINAL_FILM)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    torch.manual_seed(0)
    vit = timm.create_model(
        DINOV2_MODEL_ID, pretrained=False, num_classes=0, img_size=TINY_IMAGE_SIZE, **TINY_VIT
    ).eval()
    backbone = nn.Module()
    backbone.featurizer, backbone.use_fused_vision_backbone = vit, False
    original.FiLMedPrismaticVisionBackbone(backbone, llm_dim=FILM_DIM)
    torch.manual_seed(0)
    featurizer = PatchFeaturizer(
        DINOV2_MODEL_ID, TINY_IMAGE_SIZE, DINOV2_MEAN, DINOV2_STD, FILM_DIM, **TINY_VIT
    ).eval()
    dim = TINY_VIT["embed_dim"]
    with torch.no_grad():
        for i in range(featurizer.film.num_blocks):
            for name in ("scale", "shift"):
                ours, theirs = getattr(featurizer.film, name), getattr(vit.blocks[i], name)
                theirs.weight.copy_(ours.weight[i * dim : (i + 1) * dim])
                theirs.bias.copy_(ours.bias[i * dim : (i + 1) * dim])
    images, condition = torch.rand(2, 3, TINY_IMAGE_SIZE, TINY_IMAGE_SIZE), torch.randn(2, FILM_DIM)

    expected = vit(normalize(images), condition)

    torch.testing.assert_close(featurizer(images, condition), expected)


def test_film_conditions_every_image_of_a_sample():
    torch.manual_seed(0)
    vision = FusedVisionBackbone(TINY_IMAGE_SIZE, film_dim=FILM_DIM, **TINY_VIT).eval()
    images, condition = torch.rand(2, 2, 3, TINY_IMAGE_SIZE, TINY_IMAGE_SIZE), torch.randn(2, FILM_DIM)

    features = vision(images, condition)

    for i in range(2):
        single = vision(images[i : i + 1], condition[i : i + 1])
        torch.testing.assert_close(features[i : i + 1], single)
        assert not torch.allclose(single, vision(images[i : i + 1], torch.randn(1, FILM_DIM)))


def test_film_has_one_scale_and_shift_per_kept_block():
    vision = FusedVisionBackbone(TINY_IMAGE_SIZE, film_dim=FILM_DIM, **TINY_VIT)

    for featurizer in (vision.dinov2, vision.siglip):
        num_blocks, dim = len(featurizer.vit.blocks), featurizer.embed_dim
        assert featurizer.film.scale.weight.shape == (num_blocks * dim, FILM_DIM)
        assert featurizer.film.shift.weight.shape == (num_blocks * dim, FILM_DIM)
