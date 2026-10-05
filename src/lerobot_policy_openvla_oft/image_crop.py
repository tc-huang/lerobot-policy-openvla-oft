import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def crop_and_resize(images: Tensor, boxes: Tensor) -> Tensor:
    """Bilinearly samples each box back to the full image size, like `tf.image.crop_and_resize`.

    Args:
        images: (N, C, H, W) tensor.
        boxes: (N, 4) tensor of normalized `[y1, x1, y2, x2]` corners, where 0 and 1
            are the centers of the first and last pixel.

    Returns:
        (N, C, H, W) tensor.
    """
    height, width = images.shape[-2:]
    steps_y = torch.linspace(0, 1, height, device=images.device, dtype=images.dtype)
    steps_x = torch.linspace(0, 1, width, device=images.device, dtype=images.dtype)
    ys = boxes[:, 0:1] + (boxes[:, 2:3] - boxes[:, 0:1]) * steps_y
    xs = boxes[:, 1:2] + (boxes[:, 3:4] - boxes[:, 1:2]) * steps_x
    grid = torch.stack(torch.broadcast_tensors(xs[:, None, :], ys[:, :, None]), dim=-1)
    return F.grid_sample(images, grid * 2 - 1, mode="bilinear", align_corners=True)


def crop_boxes(num_images: int, area: float, random: bool, device: torch.device) -> Tensor:
    """Returns square boxes covering `area` of each image, centered or at a random offset.

    Random boxes reproduce the original augmentation (`dlimp.augmentations.random_resized_crop`),
    which draws the vertical and horizontal offsets from the same uniform sample, so
    a square box always moves along the image diagonal.
    """
    side = math.sqrt(area)
    if random:
        offset = torch.rand(num_images, 1, device=device) * (1 - side)
    else:
        offset = torch.full((num_images, 1), (1 - side) / 2, device=device)
    return torch.cat([offset, offset, offset + side, offset + side], dim=1)
