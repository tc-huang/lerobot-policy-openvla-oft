import timm
import torch
from tiny_models import TINY_IMAGE_SIZE, TINY_VIT

from lerobot_policy_openvla_oft.vision_backbone import (
    DINOV2_MEAN,
    DINOV2_MODEL_ID,
    DINOV2_STD,
    PatchFeaturizer,
)


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
    normalized = (images - torch.tensor(DINOV2_MEAN).view(1, 3, 1, 1)) / torch.tensor(DINOV2_STD).view(
        1, 3, 1, 1
    )

    expected = reference.get_intermediate_layers(normalized, n={TINY_VIT["depth"] - 2})[0]

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
