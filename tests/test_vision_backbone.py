import pytest
import timm
import torch

from lerobot_policy_openvla_oft.vision_backbone import (
    DINOV2_MEAN,
    DINOV2_MODEL_ID,
    DINOV2_STD,
    FusedVisionBackbone,
    PatchFeaturizer,
)

IMAGE_SIZE = 28
TINY_VIT = {"depth": 3, "embed_dim": 16, "num_heads": 2}


@pytest.fixture
def backbone():
    torch.manual_seed(0)
    return FusedVisionBackbone(IMAGE_SIZE, **TINY_VIT)


def test_output_shape_concatenates_channels_and_images(backbone):
    images = torch.rand(2, 3, 3, IMAGE_SIZE, IMAGE_SIZE)

    features = backbone(images)

    assert backbone.num_patches == (IMAGE_SIZE // 14) ** 2
    assert backbone.embed_dim == 2 * TINY_VIT["embed_dim"]
    assert features.shape == (2, 3 * backbone.num_patches, backbone.embed_dim)


def test_matches_second_to_last_block_of_unpruned_vit():
    torch.manual_seed(0)
    reference = timm.create_model(
        DINOV2_MODEL_ID, pretrained=False, num_classes=0, img_size=IMAGE_SIZE, **TINY_VIT
    ).eval()
    torch.manual_seed(0)
    featurizer = PatchFeaturizer(DINOV2_MODEL_ID, IMAGE_SIZE, DINOV2_MEAN, DINOV2_STD, **TINY_VIT).eval()
    images = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    normalized = (images - torch.tensor(DINOV2_MEAN).view(1, 3, 1, 1)) / torch.tensor(DINOV2_STD).view(
        1, 3, 1, 1
    )

    expected = reference.get_intermediate_layers(normalized, n={TINY_VIT["depth"] - 2})[0]

    torch.testing.assert_close(featurizer(images), expected)
    assert len(featurizer.vit.blocks) == TINY_VIT["depth"] - 1


def test_each_image_is_encoded_independently(backbone):
    backbone.eval()
    images = torch.rand(1, 2, 3, IMAGE_SIZE, IMAGE_SIZE)

    features = backbone(images)
    per_image = [backbone(images[:, i : i + 1]) for i in range(2)]

    torch.testing.assert_close(features, torch.cat(per_image, dim=1))


def test_every_parameter_receives_gradient(backbone):
    backbone(torch.rand(1, 1, 3, IMAGE_SIZE, IMAGE_SIZE)).sum().backward()

    assert all(p.grad is not None for p in backbone.parameters())
