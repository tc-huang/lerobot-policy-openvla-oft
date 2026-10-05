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
| Robot state input        | Yes                             | Dataset features; §3 Projectors                  |
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

### 3. Projectors

`projectors.py` holds the two MLPs that map non-text inputs into the Llama-2
token embedding space (`llm_dim = 4096`).

| Setting                       | Value                                                               | Config field            | Source                                                                            |
| ----------------------------- | ------------------------------------------------------------------- | ----------------------- | --------------------------------------------------------------------------------- |
| Vision projector              | 3-layer MLP with GELU: `2176 → 8704 → 4096 → 4096` (71M parameters) | (fixed)                 | Paper App. A, App. B.3; Repo `prismatic/extern/hf/modeling_prismatic.py:243-246`  |
| Vision projector hidden width | 4 × vision feature dim                                              | (fixed)                 | Repo `prismatic/extern/hf/modeling_prismatic.py:243`; not stated in the paper     |
| Proprio projector             | 2-layer MLP with GELU: `state_dim → 4096 → 4096`                    | (fixed)                 | Paper App. A (OFT change 2), App. B.3; Repo `prismatic/models/projectors.py:6-23` |
| Proprio projector size        | 17M parameters for the 8-dim LIBERO state                           | (derived)               | Paper Table IV                                                                    |
| Proprio tokens                | 1 token per step                                                    | (fixed)                 | Repo `prismatic/extern/hf/modeling_prismatic.py:449-459`; not stated in the paper |
| State dimension               | From the dataset's `observation.state` feature                      | (from dataset features) | Repo hardcodes `PROPRIO_DIM = 8` for LIBERO (`prismatic/vla/constants.py:29`)     |

The vision projector belongs to the pretrained OpenVLA model, while the proprio
projector is new in OpenVLA-OFT and starts from PyTorch's default
initialization (`vla-scripts/finetune.py:878-884`). How each one is trained is
covered in the LoRA fine-tuning section.

Differences from the original implementation (this port):

- **Single-purpose vision projector.** The original `PrismaticProjector` also
  supports a 2-layer variant for single (non-fused) vision backbones. OpenVLA
  always uses the fused backbone, so only the 3-layer variant is kept.
- **Proprio token shape.** `ProprioProjector` returns `(B, 1, llm_dim)`, a
  ready-to-insert token, instead of leaving the reshape to the caller.

### 4. Language model and bidirectional attention

`BidirectionalLlama` (`language_model.py`) wraps `transformers.LlamaModel` and
runs it with bidirectional self-attention. In the original autoregressive
OpenVLA, a causal mask lets each action token see only the tokens before it.
For parallel decoding, every action position must see every other one, so the
causal mask is replaced by one that only hides padding tokens.

| Setting           | Value                                                                   | Config field | Source                                                                                                                                                                          |
| ----------------- | ----------------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Language model    | Llama-2 7B: 32 layers, hidden size 4096, 32 heads, MLP size 11008, SiLU | (fixed)      | Paper App. A; Repo `llm_backbone_id: llama2-7b-pure` in the released `config.json`                                                                                              |
| Vocabulary        | 32064 (32000 Llama-2 tokens + 1 pad token, padded to a multiple of 64)  | (fixed)      | Repo: `text_config.vocab_size`, `pad_to_multiple_of` in the released `config.json`                                                                                              |
| Pad token id      | 32000                                                                   | (fixed)      | Repo: `pad_token_id` in the released `config.json`                                                                                                                              |
| RMSNorm epsilon   | 1e-6                                                                    | (fixed)      | Repo: `LlamaConfig` default, since `prismatic/extern/hf/configuration_prismatic.py:119-123` only sets the vocabulary and pad token; not stated in the paper                     |
| Attention         | Bidirectional; only padding keys are masked                             | (fixed)      | Paper App. A (OFT change 3), App. B.1; Repo `pyproject.toml:50` and transformers fork commit [`bc339d9`](https://github.com/moojink/transformers-openvla-oft/commit/bc339d9ad7) |
| Output            | Final hidden states after the last RMSNorm; no language-model head      | (fixed)      | Paper App. A (OFT change 4); Repo `prismatic/extern/hf/modeling_prismatic.py:913` (`hidden_states[-1]`)                                                                         |
| Attention backend | PyTorch SDPA                                                            | (fixed)      | This port; the fork patches the SDPA path                                                                                                                                       |

The RMSNorm epsilon of 1e-6 differs from the 1e-5 in Meta's Llama-2 7B
configuration (checked via the public mirror
[`NousResearch/Llama-2-7b-hf`](https://huggingface.co/NousResearch/Llama-2-7b-hf/blob/main/config.json),
because Meta's repository is gated). The released OpenVLA-OFT checkpoints were
fine-tuned with 1e-6, so this port keeps it.

Differences from the original implementation (this port):

- **No transformers fork.** The original depends on a fork of `transformers`
  4.40.1 whose only change rewrites Llama's causal mask inside the attention
  layer: it copies the mask's last row to every row, which for right-padded
  sequences leaves exactly the padding keys masked. This port builds that mask
  explicitly (`bidirectional_attention_mask`) and passes it to the stock
  `LlamaModel`. `transformers` 5 uses a 4D mask as given
  (`transformers/masking_utils.py`, "If the mask is already 4D, simply return
  as-is"), and the result is the same for any padding side. Tests check this
  with both the SDPA and eager attention backends.
- **No language-model head.** The original keeps `LlamaForCausalLM` and
  computes vocabulary logits that OpenVLA-OFT never uses. This port uses
  `LlamaModel`, which removes 131M parameters that would receive no gradient.
  The `lm_head` weight is dropped during conversion.

### 5. Sequence layout and parallel decoding

`OpenVLAOFT` (`model.py`) assembles the components above into one network that
predicts a whole action chunk in a single forward pass. It builds the
following input sequence for the language model:

```text
[BOS] [image patches] [proprio] [prompt] [action placeholders] [EOS] [padding]
  1     256 × images      1      varies        K × D = 56         1
```

The action placeholders are zero vectors, one per action dimension per chunk
step, so they differ only by their rotary position. Bidirectional attention
(§4) lets every placeholder read the images, the state, the prompt, and the
other placeholders. The model returns the final hidden states that decode the
56 action values; §6 maps them to actions.

| Setting                         | Value                                                                             | Config field                       | Source                                                                                                                                                                               |
| ------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Image patches and proprio token | Inserted right after BOS, proprio after the patches                               | (fixed)                            | Paper App. A; Repo `prismatic/extern/hf/modeling_prismatic.py:458`, `:475`                                                                                                           |
| Number of action placeholders   | `chunk_size × action_dim` (8 × 7 = 56 for LIBERO)                                 | `chunk_size`, action feature shape | Paper §IV-B; Repo `prismatic/extern/hf/modeling_prismatic.py:737`                                                                                                                    |
| Placeholder embeddings          | Zero vectors                                                                      | (fixed)                            | Paper §IV-B, App. B.1 ("empty action embeddings that differ only in their positional encoding"); Repo `prismatic/extern/hf/modeling_prismatic.py:621` (training), `:891` (inference) |
| EOS after the placeholders      | Llama `</s>` (id 2)                                                               | (fixed)                            | Repo: training prompt format `prismatic/models/backbones/llm/prompting/base_prompter.py:37`, inference `prismatic/extern/hf/modeling_prismatic.py:742-743`; not stated in the paper  |
| Hidden-state readout            | Shifted by one: from the last prompt token through the second-to-last placeholder | (fixed)                            | Repo: training `vla-scripts/finetune.py:343`, `:377-381`, inference `prismatic/extern/hf/modeling_prismatic.py:914`; not stated in the paper                                         |
| Padding                         | After EOS                                                                         | (fixed)                            | Repo `prismatic/util/data_utils.py:113`                                                                                                                                              |

The shifted readout is inherited from autoregressive OpenVLA, where the hidden
state at each position predicts the next token. The original training script
aligns hidden states with `labels[:, 1:]`, so the hidden state of the token
right before each action placeholder decodes that action value. The hidden
state of the last placeholder is never read. The released checkpoints were
trained this way, so this port keeps the same readout.

Differences from the original implementation (this port):

- **Placeholders without dummy tokens.** The original puts real action token
  ids (training) or the dummy id 1 (inference) into `input_ids`, embeds them,
  then multiplies by a mask derived from the labels to zero them out. This
  port inserts zero vectors directly, so it needs neither action tokens nor
  labels.
- **Batched prompts of different lengths.** The prompt arrives right-padded;
  the model appends the placeholders and EOS directly after each prompt and
  moves the padding to the end with a stable sort. Every sample therefore has
  the same layout and positions as when run alone, which the tests check.
  The original inference code only supports a batch size of 1
  (`prismatic/extern/hf/modeling_prismatic.py:747`).
- **Dependency injection.** `OpenVLAOFT` receives the vision backbone and the
  language model as constructed modules and only builds the projectors
  itself. The policy (§7) decides how to build the components from the
  configuration, and tests can pass tiny versions.
