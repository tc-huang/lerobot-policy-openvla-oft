import math

import torch

from lerobot_policy_openvla_oft.image_crop import crop_and_resize, crop_boxes


def tf_crop_and_resize(image, box):
    """Direct transcription of `tf.image.crop_and_resize` (bilinear) for one (C, H, W) image."""
    _, height, width = image.shape
    y1, x1, y2, x2 = box.tolist()
    out = torch.empty_like(image)
    for i in range(height):
        for j in range(width):
            y = (y1 + i * (y2 - y1) / (height - 1)) * (height - 1)
            x = (x1 + j * (x2 - x1) / (width - 1)) * (width - 1)
            top, left = math.floor(y), math.floor(x)
            bottom, right = min(top + 1, height - 1), min(left + 1, width - 1)
            dy, dx = y - top, x - left
            upper = image[:, top, left] * (1 - dx) + image[:, top, right] * dx
            lower = image[:, bottom, left] * (1 - dx) + image[:, bottom, right] * dx
            out[:, i, j] = upper * (1 - dy) + lower * dy
    return out


def test_matches_tensorflow_crop_and_resize():
    images = torch.rand(2, 3, 9, 9, dtype=torch.float64)
    boxes = torch.tensor([[0.1, 0.1, 0.9, 0.9], [0.05, 0.2, 0.75, 0.9]], dtype=torch.float64)

    cropped = crop_and_resize(images, boxes)

    for image, box, result in zip(images, boxes, cropped, strict=True):
        torch.testing.assert_close(result, tf_crop_and_resize(image, box))


def test_full_box_is_identity():
    images = torch.rand(1, 3, 8, 8)

    torch.testing.assert_close(crop_and_resize(images, torch.tensor([[0.0, 0.0, 1.0, 1.0]])), images)


def test_center_boxes_cover_requested_area():
    boxes = crop_boxes(2, area=0.9, random=False, device=torch.device("cpu"))
    side = math.sqrt(0.9)

    torch.testing.assert_close(
        boxes, torch.tensor([[(1 - side) / 2, (1 - side) / 2, (1 + side) / 2, (1 + side) / 2]] * 2)
    )


def test_random_boxes_move_along_the_diagonal():
    boxes = crop_boxes(1000, area=0.9, random=True, device=torch.device("cpu"))
    side = math.sqrt(0.9)

    assert torch.equal(boxes[:, 0], boxes[:, 1])
    assert (boxes[:, 0] >= 0).all() and (boxes[:, 2] <= 1).all()
    torch.testing.assert_close(boxes[:, 2] - boxes[:, 0], torch.full((1000,), side))
    assert boxes[:, 0].std() > 0
