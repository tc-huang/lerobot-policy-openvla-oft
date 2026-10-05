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
| Input images             | 1 third-person + 1 wrist camera | Dataset features; vision backbone                |
| Robot state input        | Yes                             | Dataset features; proprio projector              |
| Input image size         | 224 × 224                       | Vision backbone                                  |
| LoRA rank                | 32                              | LoRA fine-tuning                                 |
| Image augmentations      | 90% random crop, color jitter   | Processor and training                           |
| FiLM                     | No                              | Out of scope                                     |
