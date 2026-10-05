# lerobot-policy-openvla-oft

English | [繁體中文](README.zh-TW.md)

An out-of-tree [LeRobot](https://github.com/huggingface/lerobot) policy plugin
that ports [OpenVLA-OFT](https://openvla-oft.github.io/)
(Kim, Finn, and Liang, 2025) to LeRobot v0.6.1 as `--policy.type openvla_oft`.
It converts the official LIBERO checkpoints released by the authors into
LeRobot's format and evaluates them with `lerobot-eval`, aiming to reproduce
the LIBERO results reported in the paper. It also supports LoRA fine-tuning
from `openvla/openvla-7b` with `lerobot-train`, and deploying the fine-tuned
policy on a single SO-100 or SO-101 follower arm with `lerobot-rollout`.

## Design

The policy is built up one component at a time, following the paper. Each
section below describes one component and how it maps to the paper and the
original implementation.

Sources are cited as follows:

- **Paper**: Kim, Finn, and Liang,
  [Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success](https://arxiv.org/abs/2502.19645),
  arXiv:2502.19645v2.
- **Repo**: [moojink/openvla-oft](https://github.com/moojink/openvla-oft) at
  `e4287e9`, vendored as the `third_party/openvla-oft` submodule. Paths are
  relative to that directory.
- **This port**: a choice made here, not taken from either source.

### 1. Configuration

`OpenVLAOFTConfig` registers the policy type `openvla_oft` and defines the
input/output contract, normalization, and training presets. Defaults follow
the LIBERO recipe.

| Setting                        | Default                    | Config field                                      | Source                                                                                                        |
| ------------------------------ | -------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Action chunk size              | 8                          | `chunk_size`                                      | Paper §V-A, Table IV; Repo `prismatic/vla/constants.py:27`                                                    |
| Actions executed per chunk     | 8 (whole chunk, open-loop) | `n_action_steps`                                  | Paper §V-A, Table IV; Repo `experiments/robot/libero/run_libero_eval.py:100`                                  |
| Observation history            | None (single step)         | `n_obs_steps`, `observation_delta_indices`        | Paper Table IV                                                                                                |
| State and action normalization | `[q01, q99]` → `[-1, 1]`   | `normalization_mapping` (`QUANTILES`)             | Repo `prismatic/vla/constants.py:30`; the paper only states that actions are normalized to `[-1, 1]` (App. D) |
| Image normalization            | None                       | `normalization_mapping` (`IDENTITY`)              | This port: each vision backbone applies its own normalization                                                 |
| Optimizer                      | AdamW                      | `get_optimizer_preset()`                          | Repo `vla-scripts/finetune.py:935`; not stated in the paper                                                   |
| Learning rate                  | 5e-4                       | `optimizer_lr`                                    | Paper Table IV; Repo `vla-scripts/finetune.py:89`                                                             |
| Weight decay                   | 0.01                       | `optimizer_weight_decay`                          | Repo: PyTorch AdamW default, since `vla-scripts/finetune.py:935` does not set it; not stated in the paper     |
| Gradient clipping              | None                       | `optimizer_grad_clip_norm` (0)                    | Repo: `vla-scripts/finetune.py` never clips; not stated in the paper                                          |
| Learning rate decay            | ×0.1 after 100K steps      | `scheduler_decay_steps`, `scheduler_decay_factor` | Paper App. D, Table IV; Repo `vla-scripts/finetune.py:91`, `:941-944`                                         |
| Learning rate warmup           | None                       | (not supported)                                   | Repo `vla-scripts/finetune.py:90`; not stated in the paper                                                    |

The original implementation selects the chunk size and normalization scheme at
import time by inspecting the command line (`prismatic/vla/constants.py`);
here they are explicit configuration fields.

The `QUANTILES` mode in LeRobot differs from the original `BOUNDS_Q99` scheme
in two ways: the original clips normalized values to `[-1, 1]`
(`prismatic/vla/datasets/rlds/utils/data_utils.py:81`), and it leaves masked
dimensions unnormalized. For example, the LIBERO checkpoints mask the gripper
action dimension in `dataset_statistics.json`. Both differences are handled by
the processor.

The remaining hyperparameters in Paper Table IV are owned by other components
or by the training command:

| Setting (Paper Table IV) | Value                           | Handled by                                       |
| ------------------------ | ------------------------------- | ------------------------------------------------ |
| Total batch size         | 64 (8 per GPU × 8 GPUs)         | `lerobot-train --batch_size`, multi-GPU training |
| Training steps           | 150K (50K for LIBERO-Goal)      | `lerobot-train --steps`                          |
| Input images             | 1 third-person + 1 wrist camera | Dataset features; §2 Vision backbone             |
| Robot state input        | Yes                             | Dataset features; proprio projector              |
| Input image size         | 224 × 224                       | §2 Vision backbone (`image_size`)                |
| LoRA rank                | 32                              | LoRA fine-tuning                                 |
| Image augmentations      | 90% random crop, color jitter   | Processor and training                           |
| FiLM                     | No                              | Out of scope                                     |

### 2. Vision backbone

`FusedVisionBackbone` (`vision_backbone.py`) is OpenVLA's fused vision
encoder. Every camera image goes through both a DINOv2 and a SigLIP vision
transformer, built with `timm`. Each produces 256 patch features, which are
concatenated along the channel dimension (1024 + 1152 = 2176). The features of
all camera images are then concatenated along the sequence dimension, so two
cameras yield 512 tokens.

| Setting                  | Value                                                                                                                       | Config field            | Source                                                                                                                                                                                                                        |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Vision transformers      | DINOv2 ViT-L/14 with 4 registers, SigLIP SO400M/14                                                                          | (fixed)                 | Paper App. A; Repo `prismatic/extern/hf/configuration_prismatic.py:36`                                                                                                                                                        |
| Input image size         | 224 × 224                                                                                                                   | `image_size`            | Paper Table IV; Repo `prismatic/extern/hf/configuration_prismatic.py:22`                                                                                                                                                      |
| Patch features per image | 256 per transformer                                                                                                         | (derived)               | Paper App. A                                                                                                                                                                                                                  |
| Feature layer            | Output of the second-to-last block, without the final norm                                                                  | (fixed)                 | Repo `prismatic/extern/hf/modeling_prismatic.py:137`; not stated in the paper                                                                                                                                                 |
| Prefix tokens            | Dropped (DINOv2 CLS and register tokens)                                                                                    | (fixed)                 | Repo: `get_intermediate_layers` default in `modeling_prismatic.py:137`; not stated in the paper                                                                                                                               |
| Fusion                   | Concatenate DINOv2 and SigLIP features along channels                                                                       | (fixed)                 | Paper App. A; Repo `prismatic/extern/hf/modeling_prismatic.py:223`                                                                                                                                                            |
| Multiple images          | Shared backbone, concatenated along the sequence                                                                            | (from dataset features) | Paper App. A (OFT change 1); Repo `prismatic/extern/hf/modeling_prismatic.py:210-227`                                                                                                                                         |
| Pixel normalization      | DINOv2: mean `(0.484375, 0.455078125, 0.40625)`, std `(0.228515625, 0.2236328125, 0.224609375)`; SigLIP: mean and std `0.5` | (fixed)                 | Repo: `tvf_normalize_params` in the released `preprocessor_config.json`, applied by `prismatic/extern/hf/processing_prismatic.py:139` at both training (`vla-scripts/finetune.py:973`) and inference; not stated in the paper |

The DINOv2 statistics are the ImageNet mean and standard deviation rounded to
bfloat16. This rounding was baked into the released processor, so the
checkpoints were trained with these exact values.

Differences from the original implementation (this port):

- **Input layout.** The original packs each image twice into a channel-stacked
  tensor of shape `(B, 6 × num_images, H, W)`, normalized once per transformer
  by the processor. Here the backbone takes `(B, num_images, 3, H, W)` in
  `[0, 1]` and applies each transformer's normalization itself, so the
  processor stays model-agnostic.
- **Pruned layers.** The original builds the full transformers and only reads
  the second-to-last block. Here the last block, the final norm, and the
  SigLIP attention-pooling head are removed at construction. These parameters
  never affect the output, so they would receive no gradient; the original
  works around this by wrapping the model with
  `DistributedDataParallel(find_unused_parameters=True)`
  (`vla-scripts/finetune.py:875`). Removing them avoids that overhead in
  multi-GPU training. The corresponding checkpoint weights are dropped during
  conversion.
- **LayerScale naming.** The original renames timm's LayerScale `gamma`
  parameters to `scale_factor` because Hugging Face `transformers` rewrites
  parameter names that contain `gamma` (`modeling_prismatic.py:141-157`). This
  port does not load weights through `transformers`, so it keeps timm's names
  and renames the checkpoint keys during conversion.
- **Batched encoding.** All camera images are encoded in one batched call
  instead of a Python loop over images.
