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

| Setting                        | Default                                | Config field                                      | Source                                                                                                           |
| ------------------------------ | -------------------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Action chunk size              | 8                                      | `chunk_size`                                      | Paper §V-A, Table IV; Repo `prismatic/vla/constants.py:27`                                                       |
| Actions executed per chunk     | 8 (whole chunk, open-loop)             | `n_action_steps`                                  | Paper §V-A, Table IV; Repo `experiments/robot/libero/run_libero_eval.py:100`                                     |
| Observation history            | None (single step)                     | `n_obs_steps`, `observation_delta_indices`        | Paper Table IV                                                                                                   |
| State and action normalization | `[q01, q99]` → `[-1, 1]`               | `normalization_mapping` (`QUANTILES`)             | Repo `prismatic/vla/constants.py:30`; the paper only states that actions are normalized to `[-1, 1]` (App. D)    |
| Image normalization            | None                                   | `normalization_mapping` (`IDENTITY`)              | This port: each vision backbone applies its own normalization                                                    |
| Optimizer                      | AdamW                                  | `get_optimizer_preset()`                          | Repo `vla-scripts/finetune.py:935`; not stated in the paper                                                      |
| Learning rate                  | 5e-4                                   | `optimizer_lr`                                    | Paper Table IV; Repo `vla-scripts/finetune.py:89`                                                                |
| Weight decay                   | 0.01                                   | `optimizer_weight_decay`                          | Repo: PyTorch AdamW default, since `vla-scripts/finetune.py:935` does not set it; not stated in the paper        |
| Gradient clipping              | None                                   | `optimizer_grad_clip_norm` (0)                    | Repo: `vla-scripts/finetune.py` never clips; not stated in the paper                                             |
| Learning rate decay            | ×0.1 after 100K steps                  | `scheduler_decay_steps`, `scheduler_decay_factor` | Paper App. D, Table IV; Repo `vla-scripts/finetune.py:91`, `:941-944`                                            |
| Learning rate warmup           | None                                   | (not supported)                                   | Repo `vla-scripts/finetune.py:90`; not stated in the paper                                                       |
| Weight dtype                   | bfloat16                               | `dtype`                                           | Repo `vla-scripts/finetune.py:837` (model), `:895` (action head); not stated in the paper                        |
| Proprio projector weights      | float32                                | `proprio_projector_fp32`                          | Repo `vla-scripts/finetune.py:878-884` (no `to_bf16`); not stated in the paper                                   |
| Mixed precision                | Forward under bfloat16 autocast        | (follows `dtype`)                                 | Repo `vla-scripts/finetune.py:327`; not stated in the paper                                                      |
| Padded chunk steps in the loss | Included, as copies of the last action | `mask_padded_actions` (False)                     | Repo `prismatic/vla/datasets/rlds/traj_transforms.py:44`, `vla-scripts/finetune.py:390`; not stated in the paper |

The original implementation selects the chunk size and normalization scheme at
import time by inspecting the command line (`prismatic/vla/constants.py`);
here they are explicit configuration fields.

The `QUANTILES` mode in LeRobot differs from the original `BOUNDS_Q99` scheme
in two ways: the original clips normalized values to `[-1, 1]`
(`prismatic/vla/datasets/rlds/utils/data_utils.py:81`), and it leaves masked
dimensions unnormalized. For example, the LIBERO checkpoints mask the gripper
action dimension in `dataset_statistics.json`. Both differences are handled by
the processor (§8.2).

The remaining hyperparameters in Paper Table IV are owned by other components
or by the training command:

| Setting (Paper Table IV) | Value                           | Handled by                                                          |
| ------------------------ | ------------------------------- | ------------------------------------------------------------------- |
| Total batch size         | 64 (8 per GPU × 8 GPUs)         | `lerobot-train --batch_size`, multi-GPU training                    |
| Training steps           | 150K (50K for LIBERO-Goal)      | `lerobot-train --steps`                                             |
| Input images             | 1 third-person + 1 wrist camera | Dataset features; §2 Vision backbone                                |
| Robot state input        | Yes                             | Dataset features; §3 Projectors                                     |
| Input image size         | 224 × 224                       | §2 Vision backbone (`image_size`)                                   |
| LoRA rank                | 32                              | LoRA fine-tuning                                                    |
| Image augmentations      | 90% random crop, color jitter   | 90% crop: §8.3 (`image_crop_scale`); color jitter: training command |
| FiLM                     | No                              | Out of scope                                                        |

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
other placeholders. `action_hidden_states` returns the final hidden states that
decode the 56 action values; §6 maps them to actions.

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

### 6. L1 regression action head

`L1RegressionActionHead` (`action_head.py`) replaces the language model's
output layer. For each chunk step, it concatenates the hidden states of that
step's `action_dim` tokens (7 × 4096 for LIBERO) and regresses the step's
normalized action vector with a residual MLP. `OpenVLAOFT.forward` now returns
the predicted chunk of shape `(B, chunk_size, action_dim)`.

```text
LayerNorm → Linear(7·4096 → 4096) → ReLU
→ 2 × [x + ReLU(Linear(LayerNorm(x)))]
→ LayerNorm → Linear(4096 → 7)
```

| Setting         | Value                                                                                            | Config field | Source                                                                                               |
| --------------- | ------------------------------------------------------------------------------------------------ | ------------ | ---------------------------------------------------------------------------------------------------- |
| Head type       | MLP with 4 linear layers and ReLU, trained with L1 regression                                    | (fixed)      | Paper §IV-B, App. A (OFT change 4), App. B.2; Repo `prismatic/models/action_heads.py:84-107`         |
| Layer structure | Input LayerNorm and projection, 2 pre-LayerNorm residual blocks, output LayerNorm and projection | (fixed)      | Repo `prismatic/models/action_heads.py:38-81`; the paper only states "4 layers with ReLU activation" |
| Hidden width    | 4096, the language model's hidden size                                                           | (fixed)      | Repo `vla-scripts/finetune.py:894`; not stated in the paper                                          |
| Input grouping  | One step's `action_dim` hidden states, concatenated                                              | (fixed)      | Repo `prismatic/models/action_heads.py:95`, `:105`                                                   |
| Output          | Normalized actions, without squashing or clipping                                                | (fixed)      | Repo `prismatic/models/action_heads.py:81`; normalization in §1                                      |
| Head size       | 151M parameters for LIBERO                                                                       | (derived)    | Paper Table IV                                                                                       |

The L1 loss on normalized actions (Paper §IV-B, App. D; Repo
`vla-scripts/finetune.py:390`) is computed by the policy (§7).

Differences from the original implementation (this port):

- **Part of the network.** The original keeps the action head outside the
  Hugging Face model and passes it into `predict_action` at inference. Here it
  is a submodule of `OpenVLAOFT`, so a single `state_dict` holds every weight.
- **Parameter names.** `MLPResNet`'s `layer_norm1`, `fc1`,
  `mlp_resnet_blocks.N.ffn`, `layer_norm2`, and `fc2` are named `input_norm`,
  `input_proj`, `blocks.N`, `output_norm`, and `output_proj` here. Checkpoint
  keys are renamed during conversion.
- **No global chunk size.** The original reshapes with the module-level
  `NUM_ACTIONS_CHUNK` constant; this head infers the chunk length from its
  input.

### 7. Policy

`OpenVLAOFTPolicy` (`modeling_openvla_oft.py`) connects `OpenVLAOFT` to
LeRobot's `PreTrainedPolicy` interface, so `lerobot-train` and `lerobot-eval`
can use it through `--policy.type openvla_oft`. The responsibilities are split
as follows:

| Layer           | Responsibility                                                                                                                         |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Processor (§8)  | Prompt template and tokenization, image resizing and cropping, state and action normalization, action unnormalization                  |
| Policy (§7)     | Building the network from the configuration, precision, turning LeRobot batches into network inputs, the L1 loss, and the action queue |
| Network (§2–§6) | Mapping tokens, images, and state to a normalized action chunk                                                                         |

The policy reads these batch keys, all produced by the preprocessor:

| Key                                   | Shape                         | Content                                                           |
| ------------------------------------- | ----------------------------- | ----------------------------------------------------------------- |
| `observation.images.*`                | `(B, 3, H, W)` per camera     | Images in `[0, 1]`, stacked in `config.image_features` order      |
| `observation.state`                   | `(B, state_dim)`              | Normalized robot state (optional)                                 |
| `observation.language.tokens`         | `(B, L)`                      | Right-padded prompt token ids starting with BOS                   |
| `observation.language.attention_mask` | `(B, L)`                      | 1 for prompt tokens, 0 for padding                                |
| `action`                              | `(B, chunk_size, action_dim)` | Normalized target actions (training only)                         |
| `action_is_pad`                       | `(B, chunk_size)`             | Steps past the episode end (used when `mask_padded_actions=True`) |

| Behavior         | Value                                                                               | Config field                      | Source                                                                                   |
| ---------------- | ----------------------------------------------------------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------- |
| Training loss    | Mean L1 over the normalized action chunk                                            | `mask_padded_actions`             | Paper §IV-B, App. D; Repo `vla-scripts/finetune.py:390`                                  |
| Action execution | Predict a chunk, then return its first `n_action_steps` actions one per call        | `n_action_steps`                  | Paper §V-A, Table IV; Repo `experiments/robot/libero/run_libero_eval.py:306`, `:328-344` |
| Precision        | Weights in `dtype`, proprio projector optionally in float32, forward under autocast | `dtype`, `proprio_projector_fp32` | Repo, see §1                                                                             |
| Proprio input    | Used when the dataset has `observation.state`                                       | (from dataset features)           | Paper Table IV                                                                           |

Notes:

- **Precision at inference.** The original trains under autocast but runs
  evaluation in pure bfloat16 without autocast, converting the proprio
  projector and action head to bfloat16
  (`experiments/robot/openvla_utils.py:410`, `:492`). This port uses the
  training setup in both cases. Under autocast, float32 proprio weights are
  cast to bfloat16 before each matrix multiply, so they compute the same as
  bfloat16 weights; however, autocast runs some operations, such as
  LayerNorm, in float32, so evaluation numerics can differ slightly from the
  original.
- **LeRobot `use_amp`.** Leave `--policy.use_amp` disabled. The policy already
  applies autocast based on `dtype`; enabling `use_amp` would add a second
  mixed-precision layer through `accelerate`.
- **Loading memory.** The network is built in float32 and then cast to
  `dtype`, so constructing the full model briefly needs about 30 GB of host
  memory. This is revisited together with checkpoint conversion.
- **Testing.** `build_model` is the only place that knows the full-size
  architecture. Tests replace it with a tiny network, so the configuration
  contains no test-only fields.

### 8. Processor

`make_openvla_oft_pre_post_processors` (`processor_openvla_oft.py`) builds the
two LeRobot pipelines around the policy. The preprocessor turns a raw
observation (camera images, robot state, and the task string) into the batch
described in §7; the postprocessor turns the policy's normalized actions back
into robot actions. Each subsection below covers one part.

#### 8.1 Prompt and tokenization

The task description is lowercased and wrapped in OpenVLA's prompt template,
then tokenized with OpenVLA's Llama-2 tokenizer:

```text
In: What action should the robot take to {task}?\nOut: ␣
```

The template ends with a space (shown as `␣`). Llama-2's SentencePiece
tokenizer turns that trailing space into the token `▁` (id 29871). During
training, the original tokenizes the prompt and the action string together,
and this same token appears between `Out:` and the first action token. At
inference, the original builds the prompt without the space and appends id
29871 by hand. Keeping the space in the template reproduces both cases with a
plain tokenizer call.

| Setting                    | Value                                                               | Config field                            | Source                                                                                                                                                                                                  |
| -------------------------- | ------------------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Prompt template            | `In: What action should the robot take to {task}?\nOut: `           | (fixed)                                 | Repo: training `prismatic/vla/datasets/datasets.py:56` with `prismatic/models/backbones/llm/prompting/base_prompter.py:36`, inference `experiments/robot/openvla_utils.py:757`; not stated in the paper |
| Task casing                | Lowercased                                                          | (fixed)                                 | Repo: training `prismatic/vla/datasets/datasets.py:40`, inference `experiments/robot/openvla_utils.py:757`                                                                                              |
| Trailing `▁` token (29871) | Produced by the trailing space                                      | (fixed)                                 | Repo: inference appends it by hand in `prismatic/extern/hf/modeling_prismatic.py:972-975`                                                                                                               |
| Tokenizer                  | Llama-2 SentencePiece tokenizer with OpenVLA's pad token (id 32000) | `tokenizer_name` (`openvla/openvla-7b`) | Repo: `tokenizer.json` in the released checkpoints                                                                                                                                                      |
| BOS                        | Added by the tokenizer (id 1)                                       | (fixed)                                 | Repo: `add_special_tokens=True` in `prismatic/vla/datasets/datasets.py:63`                                                                                                                              |
| Padding                    | Right side, to the longest prompt in the batch                      | (fixed)                                 | This port; §5 moves padding behind the action placeholders                                                                                                                                              |

How this was verified: the prompt for a LIBERO-Spatial task was tokenized with
both the original stack (`transformers` 4.40.1) and this port's stack
(`transformers` 5.5.4). The token ids are identical, and the test
`tests/test_processor.py::test_tokens_match_original_tokenizer` pins them.
The ids also match the training sequence the original builds, up to the first
action token.

Implementation notes:

- **Two small steps.** `OpenVLAPromptProcessorStep` only formats the prompt,
  and `OpenVLATokenizerProcessorStep` only tokenizes it. Both are registered
  with LeRobot's `ProcessorStepRegistry`, so the pipeline saved next to a
  checkpoint (`policy_preprocessor.json`) records them and can be reloaded.
- **Why a custom tokenizer step.** LeRobot's `TokenizerProcessorStep` loads
  tokenizers with `AutoTokenizer`. For OpenVLA repositories, `AutoTokenizer`
  reads `config.json`, finds OpenVLA's custom model code in `auto_map`, and
  stops to ask whether to run it. The subclass loads `LlamaTokenizerFast`
  directly, which needs no custom code; everything else, including padding
  and serialization, is inherited.
- **Contract with the network.** The network (§5) requires that every prompt
  starts with BOS, is right-padded, and comes with an attention mask. The
  test `tests/test_policy.py::test_accepts_preprocessor_output` runs the real
  preprocessor output through the policy to keep both sides in agreement.

#### 8.2 State and action normalization

OpenVLA-OFT normalizes the robot state and the actions with a scheme the
original calls `BOUNDS_Q99`. For each dimension with 1st and 99th percentiles
`q01` and `q99` from the dataset statistics:

```text
normalize:    x̂ = clip(2 · (x − q01) / (q99 − q01) − 1, −1, 1)
unnormalize:  x = (x̂ + 1) / 2 · (q99 − q01) + q01
```

The pipeline builds this from LeRobot's own steps plus one small step:

```text
preprocessor:  … → normalizer_processor (QUANTILES) → openvla_oft_clip
postprocessor: unnormalizer_processor (QUANTILES) → …
```

LeRobot's `QUANTILES` mode is the first part of the formula, and
`OpenVLAClipProcessorStep` adds the clipping. Keeping LeRobot's normalizer and
unnormalizer, rather than replacing them, matters for fine-tuning:
`lerobot-train` injects the dataset statistics into a pretrained policy's
processors by looking up steps named `normalizer_processor` and
`unnormalizer_processor` (`lerobot/scripts/lerobot_train.py:357-377`). A step
under any other name would silently keep the pretrained policy's statistics.

**Unnormalized dimensions.** The original can also leave action dimensions
unnormalized: it masks the gripper of end-effector datasets, because the
gripper action is already an absolute open/close command rather than a delta.
In the released LIBERO checkpoints the gripper action is `0` (closed) to `1`
(open), the convention of the original data loader
(`experiments/robot/robot_utils.py:180-185`), and the network was trained to
output it on that raw scale. This port expresses the mask through the
statistics instead of a separate mechanism: `QUANTILES` with `q01 = -1` and
`q99 = 1` maps `x` to `2 · (x + 1) / 2 − 1 = x`, the identity. The checkpoint
conversion (§9) writes these quantiles for every dimension whose `mask` is
False in `dataset_statistics.json`.

| Setting                                     | Value                                                                                                                 | Config field                          | Source                                                                                                                                                           |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Range mapping                               | `[q01, q99]` → `[-1, 1]` for state and actions                                                                        | `normalization_mapping` (`QUANTILES`) | Repo `prismatic/vla/constants.py:30`, `prismatic/vla/datasets/rlds/utils/data_utils.py:72-83`                                                                    |
| Clipping                                    | Normalized state and action clipped to `[-1, 1]`: action targets during training, state during training and inference | (fixed)                               | Repo: training `prismatic/vla/datasets/rlds/utils/data_utils.py:81`, inference state `experiments/robot/openvla_utils.py:669-676`                                |
| Unnormalization                             | Inverse mapping without clipping                                                                                      | (fixed)                               | Repo `prismatic/extern/hf/modeling_prismatic.py:785-789`                                                                                                         |
| Masked action dimensions                    | Identity, through `q01 = -1` and `q99 = 1`                                                                            | (statistics)                          | Repo: mask for end-effector actions `prismatic/vla/datasets/rlds/oxe/materialize.py:35-39`, applied in `data_utils.py:79-83` and `modeling_prismatic.py:785-789` |
| Masked dimensions for new training          | None: every dimension uses the dataset statistics                                                                     | (statistics)                          | This port; matches the original for joint-position actions (`materialize.py:43-45`), which suits the SO-100/SO-101 arm                                           |
| Masked dimensions of the LIBERO checkpoints | The gripper                                                                                                           | (statistics)                          | Repo: `materialize.py:37-39`; recorded as `mask` in each checkpoint's `dataset_statistics.json`                                                                  |

Why the mask matters for the LIBERO checkpoints: their gripper statistics are
`q01 = 0` and `q99 = 1`. Normalizing the gripper with them would map it to
`[-1, 1]`, while the network learned to predict it on the raw `[0, 1]` scale,
so every gripper command would be misread after unnormalization.

Differences from the original implementation (this port):

- **Masked dimensions are clipped.** The original passes masked dimensions
  through without clipping; here the identity mapping is followed by the same
  clipping as every other dimension. This only changes values outside
  `[-1, 1]`, and the LIBERO gripper actions are exactly 0 or 1.
- **Epsilon.** The original always adds `1e-8` to `q99 − q01`; LeRobot only
  substitutes `1e-8` when the two are equal. The relative difference is about
  `1e-8` and has no practical effect.

How this was verified:

- `tests/test_processor.py` compares the preprocessor and postprocessor with
  direct transcriptions of the original formulas (`data_utils.py:72-83`,
  `modeling_prismatic.py:785-789`), including values outside `[q01, q99]`, and
  checks that the saved pipelines use LeRobot's `normalizer_processor` and
  `unnormalizer_processor`.
- `tests/test_convert_checkpoint.py` checks that masked gripper values pass
  through both directions unchanged.
- Against the previous implementation of this section (a custom normalizer with
  an explicit mask), the libero-spatial statistics give identical normalized
  states and unnormalized actions for 64 random inputs, including
  out-of-range values. Normalized action targets differ only where the raw
  gripper value exceeds 1, the clipping difference above.

#### 8.3 Images

Image handling is split between the preprocessor and the policy, because one
part is the same at training and inference time and the other is not:

```text
camera image ─▶ [preprocessor] resize to 224 × 224
             ─▶ [policy] crop 90% of the area and resize back to 224 × 224
                         training: square box at a random offset
                         inference: square box at the center
             ─▶ vision backbone (§2)
```

The original fine-tunes with a random crop covering 90% of the image area and
evaluates with the matching center crop, so the network always sees the same
field of view (Paper Table IV; the original `LIBERO.md:82` stresses that
evaluation must use `--center_crop True` for this reason). LeRobot runs the same preprocessor in both
modes, so the crop lives in the policy and switches on `self.training`. This
is how LeRobot's own Diffusion Policy handles its random and center crops.

| Setting        | Value                                               | Config field             | Source                                                                                                                                                                     |
| -------------- | --------------------------------------------------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Resize         | To `image_size` × `image_size`, antialiased bicubic | `image_size`             | Repo: Lanczos3 in training (`dlimp/utils.py`, `resize_image`) and evaluation (`experiments/robot/openvla_utils.py:538-541`); bicubic is this port's closest PyTorch option |
| Crop area      | 90% of the image area, square                       | `image_crop_scale` (0.9) | Paper Table IV (`random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0])`); Repo `prismatic/vla/datasets/datasets.py:152-153`                                         |
| Training crop  | Random offset, resized back with bilinear sampling  | `image_crop_scale`       | Repo: `dlimp/augmentations.py` (`random_resized_crop`), applied per camera with a different seed (`prismatic/vla/datasets/rlds/obs_transforms.py:27-38`)                   |
| Inference crop | Centered, resized back with bilinear sampling       | `image_crop_scale`       | Repo `experiments/robot/openvla_utils.py:546-593` (`crop_and_resize`), applied to every camera (`:682-708`)                                                                |
| Cameras        | Every camera image, wrist cameras included          | (fixed)                  | Repo: as above                                                                                                                                                             |

Details that matter for matching the original exactly:

- **Crop sampling.** Both original crops use `tf.image.crop_and_resize` with
  fractional box corners. `crop_and_resize` in `image_crop.py` reproduces its
  bilinear sampling with `torch.nn.functional.grid_sample`
  (`align_corners=True`); `tests/test_image_crop.py` checks it against a
  direct transcription of TensorFlow's formula.
- **Diagonal random crop.** `random_resized_crop` draws the vertical and
  horizontal offsets with the same random seed, so for a square box both
  offsets are equal and the box only moves along the image diagonal.
  `crop_boxes` reproduces this.

Known differences from the original (this port):

- **Interpolation.** The original resizes with TensorFlow's antialiased
  Lanczos3 filter, and at evaluation it first round-trips each image through
  JPEG to mimic the compressed training data. PyTorch has no Lanczos resize for
  tensors, so this port uses antialiased bicubic and skips the JPEG round trip.
  When the camera image already has the target size, the resize is skipped.
- **No 8-bit rounding.** The original converts cropped images back to 8-bit
  integers before normalization; this port keeps them in floating point.
- **Color jitter.** The original also jitters brightness, contrast,
  saturation, and hue during training (Paper Table IV). LeRobot datasets
  already provide these transforms (`--dataset.image_transforms`), so they are
  configured in the training command, not here.

These differences change pixel values only slightly. They are the first place
to look if the LIBERO success rates fall short of the paper.

### 9. Checkpoint conversion

`convert_checkpoint.py` turns a released OpenVLA-OFT checkpoint into a LeRobot
policy directory that `--policy.path` can load:

```bash
uv run python -m lerobot_policy_openvla_oft.convert_checkpoint \
    --repo-id moojink/openvla-7b-oft-finetuned-libero-spatial \
    --output-dir outputs/checkpoints/libero-spatial
```

`outputs/` is ignored by git. The script downloads only the files it needs
(about 15.5 GB per checkpoint) into the Hugging Face cache, and writes about
15 GB of bfloat16 weights plus the processor pipelines.

Released checkpoints:

| Repository                                                       | Training data           |
| ---------------------------------------------------------------- | ----------------------- |
| `moojink/openvla-7b-oft-finetuned-libero-spatial`                | LIBERO-Spatial          |
| `moojink/openvla-7b-oft-finetuned-libero-object`                 | LIBERO-Object           |
| `moojink/openvla-7b-oft-finetuned-libero-goal`                   | LIBERO-Goal             |
| `moojink/openvla-7b-oft-finetuned-libero-10`                     | LIBERO-10 (LIBERO-Long) |
| `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` | All four suites         |

What each released file becomes:

| Released file                           | Content                                                                              | Conversion                                                                                                            |
| --------------------------------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `model-0000{1..4}-of-00004.safetensors` | Vision backbone, vision projector, and Llama-2, with the LoRA weights already merged | Keys renamed as in the table below; unused weights dropped                                                            |
| `action_head--*.pt`                     | L1 regression action head (§6)                                                       | Keys renamed as in the table below                                                                                    |
| `proprio_projector--*.pt`               | Proprio projector (§3)                                                               | `module.` prefix removed                                                                                              |
| `dataset_statistics.json`               | `q01`, `q99`, and the action `mask`                                                  | LeRobot statistics for `observation.state` and `action`; masked action dimensions get `q01 = -1` and `q99 = 1` (§8.2) |
| `lora_adapter/`                         | Unmerged LoRA weights                                                                | Not needed, since the model weights already include them                                                              |

Key mapping:

| Released key                                                              | This port                                            | Reason                       |
| ------------------------------------------------------------------------- | ---------------------------------------------------- | ---------------------------- |
| `vision_backbone.featurizer.*`                                            | `model.vision.dinov2.vit.*`                          | §2                           |
| `vision_backbone.fused_featurizer.*`                                      | `model.vision.siglip.vit.*`                          | §2                           |
| `*.ls1.scale_factor`, `*.ls2.scale_factor`                                | `*.ls1.gamma`, `*.ls2.gamma`                         | timm's LayerScale names (§2) |
| `projector.*`                                                             | `model.vision_projector.*`                           | §3                           |
| `language_model.model.*`                                                  | `model.llm.model.*`                                  | §4                           |
| `module.*` (proprio projector)                                            | `model.proprio_projector.*`                          | §3                           |
| `module.model.layer_norm1`, `fc1`                                         | `model.action_head.input_norm`, `input_proj`         | §6                           |
| `module.model.mlp_resnet_blocks.N.ffn.0`, `ffn.1`                         | `model.action_head.blocks.N.norm`, `blocks.N.linear` | §6                           |
| `module.model.layer_norm2`, `fc2`                                         | `model.action_head.output_norm`, `output_proj`       | §6                           |
| `language_model.lm_head.weight`                                           | Dropped                                              | No language-model head (§4)  |
| `vision_backbone.featurizer.blocks.23.*`, `.norm.*`                       | Dropped                                              | Unused DINOv2 layers (§2)    |
| `vision_backbone.fused_featurizer.blocks.26.*`, `.norm.*`, `.attn_pool.*` | Dropped                                              | Unused SigLIP layers (§2)    |

The configuration written with each checkpoint matches LeRobot's LIBERO
environment: the third-person camera `observation.images.image` comes first
and the wrist camera `observation.images.image2` second, as in the original
(`prismatic/vla/datasets/rlds/oxe/configs.py:645-651`), followed by an 8-dim
`observation.state` and a 7-dim `action`.

How this is verified:

- **Strict loading.** `load_released_weights` maps every released key and
  loads with `strict=True`. Only the weights listed as dropped above may be
  left over; any other unmapped or missing key raises an error.
- **Offline key check.** Against the released `model.safetensors.index.json`,
  the 981 remaining keys cover all 938 vision, projector, and language-model
  parameters of this port, and the other 43 are exactly the pruned vision
  layers.
- **Tests.** `tests/test_convert_checkpoint.py` builds a released-format state
  dict from a tiny model, converts it back, and requires every value to be
  identical, plus failure cases for unknown and missing keys.
- **Real conversion.** `libero-spatial` converted on an Apple M5 Max in about
  100 seconds on the CPU, with a peak of 38 GB of host memory. Eight spot-checked
  tensors (DINOv2, SigLIP, vision projector, Llama-2, action head, proprio
  projector) are bit-identical to the released files, including the float32
  proprio projector. Reloaded with `from_pretrained`, the policy predicts a
  finite `(1, 8, 7)` action chunk in 5.7 seconds on MPS.

The released `.pt` files were saved from CUDA tensors, so the script loads them
with `map_location="cpu"`.

### 10. LIBERO evaluation

The converted checkpoints are evaluated with `lerobot-eval` in LeRobot's
built-in LIBERO environment. The table compares that environment with the
original evaluation script (`experiments/robot/libero/run_libero_eval.py` and
`libero_utils.py`) item by item.

| Item                  | Original                                                                                              | LeRobot v0.6.1                                                                                                                          | Handled by                              |
| --------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| Image rotation        | 180° (`libero_utils.py:36`, `:43`)                                                                    | 180° in `LiberoProcessorStep`                                                                                                           | LeRobot                                 |
| Cameras               | Third-person, then wrist                                                                              | `image`, then `image2`                                                                                                                  | Conversion config (§9)                  |
| State (8-dim)         | End-effector position, axis-angle orientation, gripper joint positions (`run_libero_eval.py:257-259`) | Same composition in `LiberoProcessorStep`                                                                                               | LeRobot                                 |
| Settling              | 10 no-op steps `[0, 0, 0, 0, 0, 0, -1]` (`run_libero_eval.py:318-321`)                                | Same, inside `reset()`                                                                                                                  | LeRobot                                 |
| Initial states        | The 50 fixed states per task, one per trial (`run_libero_eval.py:227-230`)                            | Each sub-environment cycles through the same fixed states                                                                               | LeRobot                                 |
| Success               | `done` from `env.step`                                                                                | `done` or `check_success()`                                                                                                             | LeRobot                                 |
| Action chunking       | Execute all 8 actions, then query again (`run_libero_eval.py:306-344`)                                | Policy action queue                                                                                                                     | Policy (§7)                             |
| **Gripper action**    | `[0, 1]` → `[-1, 1]`, binarized, negated (`run_libero_eval.py:265-274`)                               | Passed to the env unchanged                                                                                                             | **`OpenVLALiberoGripperProcessorStep`** |
| **Render resolution** | 256 × 256 (`run_libero_eval.py:116`)                                                                  | 360 × 360 by default                                                                                                                    | **Eval flags**                          |
| **Episode length**    | Spatial 220, Object 280, Goal 300, Long 520 (`run_libero_eval.py:63-69`)                              | Spatial 280, others identical                                                                                                           | **Eval flag for Spatial**               |
| **MuJoCo**            | (original setup)                                                                                      | `mujoco >= 3.4` breaks the initial states of LIBERO-Spatial task 5 ([lerobot#4390](https://github.com/huggingface/lerobot/issues/4390)) | **Eval environment**                    |
| Environment seed      | Fixed `env.seed(0)` (`libero_utils.py:24`)                                                            | Seeded from `--seed` on each reset                                                                                                      | Not matched; minor                      |

**Gripper conversion.** The original data loader stores the gripper action as
0 (close) to 1 (open), and the released checkpoints predict it on that scale
(§8.2). LIBERO expects -1 (open) or +1 (close). The original evaluation
therefore maps the prediction to `[-1, 1]`, takes its sign, and negates it
(`experiments/robot/robot_utils.py:149-198`). LeRobot only lets X-VLA inject
LIBERO-specific action processing (`lerobot/envs/factory.py`), so this port
puts the conversion into the converted checkpoints instead: `convert_checkpoint.py`
appends `OpenVLALiberoGripperProcessorStep` to their postprocessor. The step
is part of these LIBERO checkpoints, not of the policy; a policy trained on
data that already uses LIBERO's convention does not need it.

**Evaluation command.** One suite, matching the original protocol:

```bash
lerobot-eval \
    --policy.path=outputs/checkpoints/libero-spatial \
    --policy.device=cuda \
    --env.type=libero \
    --env.task=libero_spatial \
    --env.observation_height=256 \
    --env.observation_width=256 \
    --env.episode_length=220 \
    --eval.n_episodes=50 \
    --eval.batch_size=10 \
    --seed=7
```

`--env.episode_length` is only needed for LIBERO-Spatial; the other suites
already use the original limits. `--eval.n_episodes=50` runs each of the 10
tasks of a suite 50 times, 500 trials per suite like the paper.

**Evaluation environment.** LeRobot's `libero` extra installs only on Linux
(`hf-libero` is marked `sys_platform == 'linux'`), so evaluation needs a Linux
machine. Pin `mujoco<3.4` in that environment because of lerobot#4390.

**Targets.** Success rates reported in the paper, each averaged over 500 trials
per suite and, per the original `LIBERO.md`, over three seeds:

| Checkpoint                     | Spatial | Object | Goal | Long | Average | Source          |
| ------------------------------ | ------- | ------ | ---- | ---- | ------- | --------------- |
| One policy per suite           | 97.6    | 98.4   | 97.9 | 94.5 | 97.1    | Paper Table I   |
| One policy for all four suites | 97.7    | 98.0   | 96.1 | 95.3 | 96.8    | Paper Table XIV |

How close this port must come is decided once results are available. The
known sources of difference are listed in §7 (inference precision) and §8.3
(image interpolation).

How this is verified: `tests/test_processor.py` compares
`OpenVLALiberoGripperProcessorStep` with a transcription of the original
`normalize_gripper_action` and `invert_gripper_action`, and
`tests/test_convert_checkpoint.py` checks that the LIBERO postprocessor applies
it after unnormalization.

#### Results

All four suites were evaluated with the released single-suite checkpoints, one
seed each (`--seed=7`, 500 episodes per suite). The paper averages three seeds.

| Suite          | Checkpoint       | This port (%) | Paper Table I (%) | Difference |
| -------------- | ---------------- | ------------- | ----------------- | ---------- |
| LIBERO-Spatial | `libero-spatial` | 98.0          | 97.6              | +0.4       |
| LIBERO-Object  | `libero-object`  | 98.0          | 98.4              | -0.4       |
| LIBERO-Goal    | `libero-goal`    | 97.2          | 97.9              | -0.7       |
| LIBERO-Long    | `libero-10`      | 94.2          | 94.5              | -0.3       |
| **Average**    |                  | **96.85**     | **97.1**          | **-0.25**  |

With 500 episodes, the 95% confidence interval of a single success rate near
97% is about ±1.5 points, so every suite is within the sampling error of the
paper.

**LIBERO-Spatial** (`moojink/openvla-7b-oft-finetuned-libero-spatial`)

| Task | Instruction                                                                              | Success             |
| ---- | ---------------------------------------------------------------------------------------- | ------------------- |
| 0    | pick up the black bowl between the plate and the ramekin and place it on the plate       | 50/50               |
| 1    | pick up the black bowl next to the ramekin and place it on the plate                     | 49/50               |
| 2    | pick up the black bowl from table center and place it on the plate                       | 50/50               |
| 3    | pick up the black bowl on the cookie box and place it on the plate                       | 50/50               |
| 4    | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 47/50               |
| 5    | pick up the black bowl on the ramekin and place it on the plate                          | 45/50               |
| 6    | pick up the black bowl next to the cookie box and place it on the plate                  | 50/50               |
| 7    | pick up the black bowl on the stove and place it on the plate                            | 49/50               |
| 8    | pick up the black bowl next to the plate and place it on the plate                       | 50/50               |
| 9    | pick up the black bowl on the wooden cabinet and place it on the plate                   | 50/50               |
|      | **Total**                                                                                | **490/500 (98.0%)** |

**LIBERO-Object** (`moojink/openvla-7b-oft-finetuned-libero-object`)

| Task | Instruction                                              | Success             |
| ---- | -------------------------------------------------------- | ------------------- |
| 0    | pick up the alphabet soup and place it in the basket     | 49/50               |
| 1    | pick up the cream cheese and place it in the basket      | 50/50               |
| 2    | pick up the salad dressing and place it in the basket    | 49/50               |
| 3    | pick up the bbq sauce and place it in the basket         | 49/50               |
| 4    | pick up the ketchup and place it in the basket           | 50/50               |
| 5    | pick up the tomato sauce and place it in the basket      | 50/50               |
| 6    | pick up the butter and place it in the basket            | 49/50               |
| 7    | pick up the milk and place it in the basket              | 48/50               |
| 8    | pick up the chocolate pudding and place it in the basket | 46/50               |
| 9    | pick up the orange juice and place it in the basket      | 50/50               |
|      | **Total**                                                | **490/500 (98.0%)** |

**LIBERO-Goal** (`moojink/openvla-7b-oft-finetuned-libero-goal`)

| Task | Instruction                                 | Success             |
| ---- | ------------------------------------------- | ------------------- |
| 0    | open the middle drawer of the cabinet       | 49/50               |
| 1    | put the bowl on the stove                   | 47/50               |
| 2    | put the wine bottle on top of the cabinet   | 47/50               |
| 3    | open the top drawer and put the bowl inside | 43/50               |
| 4    | put the bowl on top of the cabinet          | 50/50               |
| 5    | push the plate to the front of the stove    | 50/50               |
| 6    | put the cream cheese in the bowl            | 50/50               |
| 7    | turn on the stove                           | 50/50               |
| 8    | put the bowl on the plate                   | 50/50               |
| 9    | put the wine bottle on the rack             | 50/50               |
|      | **Total**                                   | **486/500 (97.2%)** |

**LIBERO-Long** (`moojink/openvla-7b-oft-finetuned-libero-10`)

| Task | Instruction                                                                             | Success             |
| ---- | --------------------------------------------------------------------------------------- | ------------------- |
| 0    | put both the alphabet soup and the tomato sauce in the basket                           | 46/50               |
| 1    | put both the cream cheese box and the butter in the basket                              | 49/50               |
| 2    | turn on the stove and put the moka pot on it                                            | 50/50               |
| 3    | put the black bowl in the bottom drawer of the cabinet and close it                     | 48/50               |
| 4    | put the white mug on the left plate and put the yellow and white mug on the right plate | 47/50               |
| 5    | pick up the book and place it in the back compartment of the caddy                      | 50/50               |
| 6    | put the white mug on the plate and put the chocolate pudding to the right of the plate  | 45/50               |
| 7    | put both the alphabet soup and the cream cheese box in the basket                       | 49/50               |
| 8    | put both moka pots on the stove                                                         | 39/50               |
| 9    | put the yellow and white mug in the microwave and close it                              | 48/50               |
|      | **Total**                                                                               | **471/500 (94.2%)** |

Run details (2026-10-05):

| Item       | Value                                                                                                                                                                                   |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Commands   | `sky launch -c openvla-oft-eval skypilot/libero_eval.yaml -i 15 --down` with a 2-episode smoke test, then one `sky exec` per suite with `CHECKPOINT`, `SUITE`, and `EPISODE_LENGTH` set |
| Code       | `db72470` (Spatial), `857fd66` (Object, Goal, Long; only the SkyPilot task changed)                                                                                                     |
| Machine    | RunPod L40S (48 GB), 12 vCPUs, 62 GB RAM, region EU-NL-1, $1.09/hour list price; no L40 was available in any region                                                                     |
| Software   | Python 3.12, PyTorch 2.11.0+cu130, transformers 5.5.4, LeRobot 0.6.1, MuJoCo 3.3.7, robosuite 1.4.0, `MUJOCO_GL=egl`                                                                    |
| Evaluation | 10 environments in parallel (`--eval.batch_size=10`), 256 × 256 rendering, original episode limits                                                                                      |

| Suite          | Episode limit | Evaluation time | Per episode (10 in parallel) |
| -------------- | ------------- | --------------- | ---------------------------- |
| LIBERO-Spatial | 220           | 26.1 min        | 3.1 s                        |
| LIBERO-Object  | 280           | 30.8 min        | 3.7 s                        |
| LIBERO-Goal    | 300           | 27.5 min        | 3.3 s                        |
| LIBERO-Long    | 520           | 57.5 min        | 6.9 s                        |

| Stage                                                                  | Time                                                                                                                                                                                    |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Provisioning the L40S                                                  | About 2 minutes                                                                                                                                                                         |
| Setup on a fresh machine (`apt-get`, `uv sync --extra libero`)         | About 1 minute                                                                                                                                                                          |
| Checkpoint download from the Hugging Face Hub (15.5 GB)                | About 1 to 2 minutes per checkpoint                                                                                                                                                     |
| Checkpoint conversion                                                  | About 2 minutes per checkpoint                                                                                                                                                          |
| Whole job per suite, including download, conversion, and model loading | Spatial 28.3 min, Object 35.4 min, Goal 32.0 min, Long 62.0 min                                                                                                                         |
| Total machine time                                                     | About 3 hours 4 minutes (15:30 to 18:34 UTC), including two setup fixes and a 2-episode smoke test; about $3.3 at the list price. The cluster was torn down right after the last suite. |

The lowest tasks are LIBERO-Long task 8 (78%, two moka pots on the stove),
LIBERO-Goal task 3 (86%), and LIBERO-Spatial task 5 (90%). Spatial task 5 is
the one affected by lerobot#4390; this run used MuJoCo 3.3.7, below the
affected 3.4 range.

### 11. LoRA fine-tuning

Fine-tuning follows LeRobot's own PEFT workflow
([PEFT training guide](https://huggingface.co/docs/lerobot/peft_training)):
`lerobot-train` starts from a pretrained LeRobot policy (`--policy.path`), wraps
it with LoRA adapters (`--peft.*`), and saves only the adapters and the fully
trained modules. For OpenVLA-OFT, the pretrained policy is the base OpenVLA
model converted by `convert_checkpoint.py --base`.

```bash
# 1. Convert the base OpenVLA model once (about 15 GB).
uv run python -m lerobot_policy_openvla_oft.convert_checkpoint --base \\
    --repo-id openvla/openvla-7b \\
    --output-dir outputs/checkpoints/openvla-7b

# 2. Fine-tune with LoRA on 8 GPUs (8 per GPU, 64 in total).
uv run accelerate launch --multi_gpu --num_processes=8 $(uv run which lerobot-train) \\
    --policy.path=outputs/checkpoints/openvla-7b \\
    --policy.push_to_hub=false \\
    --peft.method_type=LORA \\
    --peft.r=32 \\
    --dataset.repo_id=lerobot/libero \\
    --dataset.image_transforms.enable=true \\
    --dataset.image_transforms.max_num_transforms=4 \\
    --dataset.image_transforms.tfs='{"brightness": {"type": "ColorJitter", "kwargs": {"brightness": [0.8, 1.2]}}, "contrast": {"type": "ColorJitter", "kwargs": {"contrast": [0.8, 1.2]}}, "saturation": {"type": "ColorJitter", "kwargs": {"saturation": [0.8, 1.2]}}, "hue": {"type": "ColorJitter", "kwargs": {"hue": [-0.05, 0.05]}}}' \\
    --batch_size=8 \\
    --steps=150000 \\
    --output_dir=outputs/train/openvla-oft-libero
```

| Setting                 | Value                                                                                                      | How it is set                                   | Source                                                                        |
| ----------------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------- |
| Starting point          | `openvla/openvla-7b`                                                                                       | `--policy.path` to the converted base           | Repo `vla-scripts/finetune.py:71` (`vla_path` default), loaded at `:835-837`  |
| LoRA rank               | 32                                                                                                         | `--peft.r=32`                                   | Paper Table IV; Repo `vla-scripts/finetune.py:107`                            |
| LoRA alpha              | 16                                                                                                         | Policy default                                  | Repo `vla-scripts/finetune.py:849` (`min(rank, 16)`); not stated in the paper |
| LoRA dropout            | 0                                                                                                          | Policy default                                  | Repo `vla-scripts/finetune.py:108`; not stated in the paper                   |
| LoRA initialization     | Gaussian                                                                                                   | Policy default                                  | Repo `vla-scripts/finetune.py:852`; not stated in the paper                   |
| LoRA targets            | Every linear layer of the vision backbone, vision projector, and language model                            | Policy default (`LORA_TARGET_MODULES`)          | Repo `vla-scripts/finetune.py:851` (`all-linear`)                             |
| Fully trained modules   | Action head (151M) and proprio projector (17M)                                                             | Policy default (`modules_to_save`)              | Paper Table IV; Repo `vla-scripts/finetune.py:876-896`, `:927-933`            |
| Batch size              | 8 per GPU, 64 in total on 8 GPUs                                                                           | `--batch_size=8` with 8 processes               | Paper Table IV; Repo `vla-scripts/finetune.py:88`                             |
| Learning rate and decay | 5e-4, ×0.1 after 100K steps                                                                                | Policy presets (§1)                             | Paper Table IV, App. D                                                        |
| Training steps          | 150K (50K for LIBERO-Goal)                                                                                 | `--steps`                                       | Paper Table IV                                                                |
| Random crop             | 90% of the area                                                                                            | Policy (§8.3)                                   | Paper Table IV                                                                |
| Color jitter            | Brightness, contrast `[0.8, 1.2]`, saturation `[0.8, 1.2]`, hue `[-0.05, 0.05]`, all applied in this order | `--dataset.image_transforms.*`                  | Paper Table IV; Repo `prismatic/vla/datasets/datasets.py:152-165`             |
| Mixed precision         | bfloat16 autocast from the policy                                                                          | Do not pass `--mixed_precision` to `accelerate` | §7                                                                            |

Why LoRA counts match the paper: rank 32 on every targeted layer gives 107.9M
adapter parameters. The original applies `all-linear` to the full model,
including the vision layers this port removes (§2) and the language-model head
(§4); counting those gives 110.8M, the paper's "111M LoRA adapter" in Table IV.
`tests/test_peft.py` checks that `LORA_TARGET_MODULES` selects exactly the
linear layers of the full-size network.

How the pieces fit LeRobot:

- **Base checkpoint without features.** The converted base has empty
  `input_features` and `output_features`, so `lerobot-train` takes them from the
  dataset. It holds no action head or proprio projector; LeRobot loads
  checkpoints non-strictly, so these keep their PyTorch initialization, as in
  the original (`vla-scripts/finetune.py:876-896`).
- **Dataset statistics.** The processors saved with the base contain LeRobot's
  `normalizer_processor` and `unnormalizer_processor` (§8.2), into which
  `lerobot-train` injects the dataset statistics.
- **PEFT defaults.** `OpenVLAOFTPolicy._get_default_peft_targets()` supplies the
  targets, the fully trained modules, alpha, dropout, and initialization.
  LeRobot's CLI always passes `--peft.r` (default 16), so the rank has to be set
  on the command line.
- **Checkpoints.** Each checkpoint holds the LoRA adapters plus the action head
  and proprio projector, and its adapter config points back to the base
  directory. To evaluate a fine-tuned policy, pass its checkpoint as
  `--policy.path`; LeRobot loads the base from that recorded path, so the base
  directory must exist at the same location.

Differences from the original implementation (this port):

- **Brightness jitter.** The original adds a random offset in `[-0.2, 0.2]`
  (`tf.image.stateless_random_brightness` in `dlimp/augmentations.py`);
  torchvision's `ColorJitter` multiplies by a factor in `[0.8, 1.2]` instead.
- **Augmentation order.** The original crops before the color jitter; here the
  dataset applies the jitter and the policy crops afterwards. The color
  transforms act on every pixel alike, so the order only matters through
  clipping at 0 and 1.
- **Unmerged adapters.** The original merges the LoRA weights into the model at
  every checkpoint (`vla-scripts/finetune.py:653-660`); LeRobot keeps them
  separate.
- **Training data.** `lerobot/libero` contains the four LIBERO suites in
  LeRobot's format. Whether its demonstrations match the original's filtered
  `*_no_noops` RLDS datasets has not been checked yet.

How this was verified: `tests/test_peft.py` runs the `lerobot-train` wrapping
path on a tiny policy and checks that only the LoRA adapters, the action head,
and the proprio projector are trainable; that the original alpha, dropout, and
initialization are used; that an adapter survives saving and reloading; and
that a base policy loads the VLA weights exactly while initializing the new
modules. A real training run has not been done yet.
