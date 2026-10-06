from typing import Any

import timm
import torch
from timm.models.vision_transformer import Block, VisionTransformer
from torch import Tensor, nn

DINOV2_MODEL_ID = "vit_large_patch14_reg4_dinov2.lvd142m"
SIGLIP_MODEL_ID = "vit_so400m_patch14_siglip_224"

# Values from the released `preprocessor_config.json`; the DINOv2 ImageNet
# statistics were rounded to bfloat16 when the original processor was exported.
DINOV2_MEAN = (0.484375, 0.455078125, 0.40625)
DINOV2_STD = (0.228515625, 0.2236328125, 0.224609375)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


class FiLMGenerator(nn.Module):
    """Projects a language embedding to FiLM scale and shift vectors for every ViT block.

    Each block has its own affine projections, stored as slices of one linear layer
    per vector so that all blocks are computed at once.
    """

    def __init__(self, condition_dim: int, num_blocks: int, feature_dim: int):
        super().__init__()
        self.num_blocks = num_blocks
        self.scale = nn.Linear(condition_dim, num_blocks * feature_dim)
        self.shift = nn.Linear(condition_dim, num_blocks * feature_dim)

    def forward(self, condition: Tensor) -> tuple[Tensor, Tensor]:
        """Maps (B, condition_dim) to `gamma` and `beta`, each (num_blocks, B, 1, feature_dim)."""
        gamma, beta = (
            proj(condition).unflatten(-1, (self.num_blocks, 1, -1)) for proj in (self.scale, self.shift)
        )
        return gamma.movedim(1, 0), beta.movedim(1, 0)


def film_block(block: Block, x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
    """Runs a timm ViT block with FiLM between its attention and MLP sublayers."""
    x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
    x = x * (1 + gamma) + beta
    return x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))


class PatchFeaturizer(nn.Module):
    """Normalizes an image and returns the patch tokens of a ViT's second-to-last block.

    With `film_dim`, every block is modulated by FiLM conditioned on a
    `film_dim`-dimensional vector.
    """

    def __init__(
        self,
        model_id: str,
        image_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        film_dim: int | None = None,
        **vit_overrides: Any,
    ):
        super().__init__()
        self.vit: VisionTransformer = timm.create_model(
            model_id, pretrained=False, num_classes=0, img_size=image_size, **vit_overrides
        )
        self.vit.prune_intermediate_layers([len(self.vit.blocks) - 2], prune_norm=True, prune_head=True)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.film = FiLMGenerator(film_dim, len(self.vit.blocks), self.vit.embed_dim) if film_dim else None

    @property
    def embed_dim(self) -> int:
        return self.vit.embed_dim

    @property
    def num_patches(self) -> int:
        return self.vit.patch_embed.num_patches

    def forward(self, images: Tensor, condition: Tensor | None = None) -> Tensor:
        """Maps images (B, 3, H, W) in [0, 1] to patch features (B, num_patches, embed_dim).

        `condition` is the (B, film_dim) FiLM input, required if FiLM is enabled.
        """
        x = (images - self.mean) / self.std
        if self.film is None:
            tokens = self.vit.forward_features(x)
        else:
            vit = self.vit
            tokens = vit.norm_pre(vit.patch_drop(vit._pos_embed(vit.patch_embed(x))))
            for block, gamma, beta in zip(vit.blocks, *self.film(condition), strict=True):
                tokens = film_block(block, tokens, gamma, beta)
        return tokens[:, self.vit.num_prefix_tokens :]


class FusedVisionBackbone(nn.Module):
    """OpenVLA's fused DINOv2 + SigLIP vision encoder.

    Each camera image is encoded by both ViTs; their patch features are concatenated
    along the channel dimension, and the patches of all cameras are concatenated
    along the sequence dimension. With `film_dim`, both ViTs use FiLM (OpenVLA-OFT+).
    """

    def __init__(self, image_size: int, film_dim: int | None = None, **vit_overrides: Any):
        super().__init__()
        self.dinov2 = PatchFeaturizer(
            DINOV2_MODEL_ID, image_size, DINOV2_MEAN, DINOV2_STD, film_dim, **vit_overrides
        )
        self.siglip = PatchFeaturizer(
            SIGLIP_MODEL_ID, image_size, SIGLIP_MEAN, SIGLIP_STD, film_dim, **vit_overrides
        )

    @property
    def uses_film(self) -> bool:
        return self.dinov2.film is not None

    @property
    def embed_dim(self) -> int:
        return self.dinov2.embed_dim + self.siglip.embed_dim

    @property
    def num_patches(self) -> int:
        """Number of patch features produced per image."""
        return self.dinov2.num_patches

    def forward(self, images: Tensor, condition: Tensor | None = None) -> Tensor:
        """Encodes multi-camera images.

        Args:
            images: (B, num_images, 3, H, W) tensor with values in [0, 1].
            condition: (B, film_dim) FiLM input shared by all images of a sample,
                required if FiLM is enabled.

        Returns:
            (B, num_images * num_patches, embed_dim) patch features.
        """
        batch_size, num_images = images.shape[:2]
        flat = images.flatten(0, 1)
        if condition is not None:
            condition = condition.repeat_interleave(num_images, dim=0)
        features = torch.cat([self.dinov2(flat, condition), self.siglip(flat, condition)], dim=-1)
        return features.reshape(batch_size, -1, self.embed_dim)
