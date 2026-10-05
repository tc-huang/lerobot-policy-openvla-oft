from typing import Any

import timm
import torch
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor, nn

DINOV2_MODEL_ID = "vit_large_patch14_reg4_dinov2.lvd142m"
SIGLIP_MODEL_ID = "vit_so400m_patch14_siglip_224"

# Values from the released `preprocessor_config.json`; the DINOv2 ImageNet
# statistics were rounded to bfloat16 when the original processor was exported.
DINOV2_MEAN = (0.484375, 0.455078125, 0.40625)
DINOV2_STD = (0.228515625, 0.2236328125, 0.224609375)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


class PatchFeaturizer(nn.Module):
    """Normalizes an image and returns the patch tokens of a ViT's second-to-last block."""

    def __init__(
        self,
        model_id: str,
        image_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        **vit_overrides: Any,
    ):
        super().__init__()
        self.vit: VisionTransformer = timm.create_model(
            model_id, pretrained=False, num_classes=0, img_size=image_size, **vit_overrides
        )
        self.vit.prune_intermediate_layers([len(self.vit.blocks) - 2], prune_norm=True, prune_head=True)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    @property
    def embed_dim(self) -> int:
        return self.vit.embed_dim

    @property
    def num_patches(self) -> int:
        return self.vit.patch_embed.num_patches

    def forward(self, images: Tensor) -> Tensor:
        """Maps images (B, 3, H, W) in [0, 1] to patch features (B, num_patches, embed_dim)."""
        tokens = self.vit.forward_features((images - self.mean) / self.std)
        return tokens[:, self.vit.num_prefix_tokens :]


class FusedVisionBackbone(nn.Module):
    """OpenVLA's fused DINOv2 + SigLIP vision encoder.

    Each camera image is encoded by both ViTs; their patch features are concatenated
    along the channel dimension, and the patches of all cameras are concatenated
    along the sequence dimension.
    """

    def __init__(self, image_size: int, **vit_overrides: Any):
        super().__init__()
        self.dinov2 = PatchFeaturizer(DINOV2_MODEL_ID, image_size, DINOV2_MEAN, DINOV2_STD, **vit_overrides)
        self.siglip = PatchFeaturizer(SIGLIP_MODEL_ID, image_size, SIGLIP_MEAN, SIGLIP_STD, **vit_overrides)

    @property
    def embed_dim(self) -> int:
        return self.dinov2.embed_dim + self.siglip.embed_dim

    @property
    def num_patches(self) -> int:
        """Number of patch features produced per image."""
        return self.dinov2.num_patches

    def forward(self, images: Tensor) -> Tensor:
        """Encodes multi-camera images.

        Args:
            images: (B, num_images, 3, H, W) tensor with values in [0, 1].

        Returns:
            (B, num_images * num_patches, embed_dim) patch features.
        """
        batch_size = images.shape[0]
        flat = images.flatten(0, 1)
        features = torch.cat([self.dinov2(flat), self.siglip(flat)], dim=-1)
        return features.reshape(batch_size, -1, self.embed_dim)
