# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repo is a **LeRobot out-of-tree policy plugin** (distribution name `lerobot_policy_openvla_oft`).
The goal is to **rewrite OpenVLA-OFT the LeRobot-native way**, load the
`moojink/openvla-7b-oft-finetuned-libero-spatial` weights, and verify the LIBERO-Spatial
success rate (paper reports ~97%) via `lerobot-eval --env.type=libero`.

**Scope: inference + LIBERO eval only.** `forward()` (training) deliberately stays
`raise NotImplementedError` for this round.

The full roadmap lives in `plan.md`; per-item progress in `progress.md` (both in Traditional
Chinese). These two are the authoritative working docs for this repo.
**Before starting work, read the relevant milestone in `plan.md`; after finishing, update
`progress.md`.**

## Current state (important)

`src/lerobot_policy_openvla_oft/` holds three stub files whose main methods still
`raise NotImplementedError`:
- `configuration_openvla_oft.py` — `OpenvlaOftConfig`
- `modeling_openvla_oft.py` — `OpenvlaOftPolicy`
- `processor_openvla_oft.py` — `make_openvla_oft_pre_post_processors`

Class names already match the factory derivation rules (see below). **M0 is mostly done**:
`[build-system]` is in `pyproject.toml`, `__init__.py` exports the correct names, lerobot is aligned
to 0.5.2, and `register_third_party_plugins()` discovers the plugin. Still pending in M0: add the
`transformers` / `timm` / `tokenizers` deps and confirm the installed lerobot has `LiberoProcessorStep`.

The full upstream `prismatic` package has been **vendored** into
`src/lerobot_policy_openvla_oft/prismatic/` — this is the working source for M1 (see the two-phase
prismatic strategy under "Key constraints"). Read the relevant milestone in `plan.md` before working.

## Commands

```bash
uv sync                      # Install dependencies (incl. dev group), pinned by uv.lock
uv pip install -e .          # Editable install as a distribution (needs [build-system] in pyproject first)
uv run pytest                # Run tests
uv run pytest tests/test_x.py::test_name   # Run a single test
uv run ruff format .         # Format
uv run ruff check --fix .    # Lint + autofix
uv run mypy src              # Type check
pre-commit run --all-files   # Run all hooks (ruff/typos/pyupgrade/prettier/gitleaks/zizmor/bandit/mypy)
```

End-to-end eval (dev path; requires the plugin installed + LIBERO simulator, and CUDA in practice):

```bash
lerobot-eval --policy.type=openvla_oft \
  --policy.pretrained_oft_repo=moojink/openvla-7b-oft-finetuned-libero-spatial \
  --env.type=libero --env.task=libero_spatial --eval.n_episodes=50
```

Environment: Python 3.12, managed by uv, src layout. Local Mac has no CUDA (MPS/CPU can only run
the correctness gate; full eval needs a GPU box).

## Architecture: the LeRobot plugin contract (the part that needs reading multiple files)

LeRobot wires up third-party policies through **naming conventions + dynamic imports**. The three
file names and class names are **not arbitrary**: the factory derives the other two from the config
class name and module name, so a naming mismatch is a runtime `AttributeError`:

- **Discovery**: `register_third_party_plugins()` in `references/lerobot/.../utils/import_utils.py`
  calls `importlib.import_module("lerobot_policy_openvla_oft")` directly, so the distribution name
  must be the importable underscore name.
- **Policy derivation** (`policies/factory.py`): strip `Config` from the config class name and append
  `Policy` → `OpenvlaOftConfig` yields `OpenvlaOftPolicy`; the module is derived by replacing
  `configuration_` with `modeling_`.
- **Processor derivation**: from `config.type` (the registration string `openvla_oft`) it builds
  `make_openvla_oft_pre_post_processors`; the module is derived by replacing `configuration_` with
  `processor_`.
- The registration string `openvla_oft` comes from `@PreTrainedConfig.register_subclass("openvla_oft")`.

**Before renaming anything, confirm this chain stays self-consistent.** The current names are correct;
mirroring `pi0` / `act` / `smolvla` under `references/lerobot` is the safest approach.

### Normalization lives entirely in the processor, not the policy

`PreTrainedPolicy` (`pretrained.py`) does no normalization itself. Normalization is handled by the
processor pipeline's `NormalizerProcessorStep` / `UnnormalizerProcessorStep` (built from
`features + norm_map + dataset_stats`). The model only emits/consumes **normalized** actions;
unnormalization is the postprocessor's job. LeRobot's `NormalizationMode.QUANTILES` (q01/q99 mapped to
[-1, 1]) is equivalent to OFT's `BOUNDS_Q99`.

### Rollout wiring (at eval time)

`env_preprocessor (LiberoProcessorStep) → policy preprocessor → policy.select_action → policy postprocessor`.
When the action queue is empty, `select_action` calls `predict_action_chunk`, takes the first
`n_action_steps` into a deque, and `popleft()`s one per step (copy the queue pattern from
`act/modeling_act.py` and `pi0/modeling_pi0.py`).

## Key constraints and pitfalls

- **Vendored `prismatic`, two-phase strategy**: the full upstream `prismatic` package is vendored into
  `src/lerobot_policy_openvla_oft/prismatic/`. The inference-critical pieces are
  `OpenVLAForActionPrediction` (`extern/hf/modeling_prismatic.py:720`), `L1RegressionActionHead`
  (`models/action_heads.py:84`), and `ProprioProjector` (`models/projectors.py`).
  **Phase A (current M1)** — reuse the vendored copy in place: delete files/branches the inference path
  does not need (RLDS/OXE data pipeline, FSDP/DDP training, native loading path, FiLM, diffusion) and
  clear the import blockers below. **Phase B (after the M5 gate, non-blocking)** — gradually move the
  kept files into a clean `model/` submodule and cut the dead diffusion/FiLM branches. Both phases
  reuse the moojink pretrained weights, do **not** import the upstream installed `prismatic` package
  root, and must keep `sys.modules` free of `tensorflow` / `dlimp` / `diffusers`.
- **Import blockers to clear before wiring** (verified by reading the vendored source): (1) 36 of 66
  files use absolute `from prismatic.…` imports — rewrite the kept files to
  `from lerobot_policy_openvla_oft.prismatic.…`; (2) `action_heads.py:7` imports `diffusers` at module
  top (same file as `L1RegressionActionHead`) — drop the diffusers import and the diffusion classes,
  keeping `MLPResNet` + `L1RegressionActionHead`; (3) `vla/constants.py` runs `sys.argv` platform
  detection and prints at import — hard-code the LIBERO constants. The package `__init__.py` files are
  already emptied (the eager import chain is broken), and `tensorflow/dlimp` only live under
  `vla/datasets/rlds/*`, which the inference path never touches.
- **v1 scope trim**: LIBERO-Spatial uses `use_l1_regression=True, use_diffusion=False, use_film=False,
  num_images_in_input=2, use_proprio=True, chunk=8, action_dim=7`.
  **Do not port FiLM, do not port diffusion / noisy action projector, no diffusers needed.**
- **Do not redo LIBERO observation handling**: `LiberoProcessorStep` in `processor/env_processor.py`
  already does the 180° image flip and assembles the 8-dim proprio
  (`eef_pos(3) + quat2axisangle(3) + gripper_qpos(2)`), identical to OFT. Image keys are
  `observation.images.image` (agentview) and `observation.images.image2` (eye_in_hand).
- **transformers version sensitivity**: the Llama-2 parallel-decoding path runs a single forward on
  `inputs_embeds` + a 2D `attention_mask` (Llama builds the causal mask internally — there is **no**
  custom 4D mask; "parallel decoding" just means inserting all action placeholder tokens at once and
  reading the action-position hidden states with the L1 head). Its handling of `inputs_embeds` /
  `position_ids=None` / legacy `past_key_values` is version-sensitive; stock 5.x is incompatible. Pin a
  working version (moojink fork 4.40.1, or an empirically working one) and isolate it from the lerobot
  base (which does not depend on transformers).
- **Numerical consistency is the biggest risk**: image preprocessing (lanczos resize, 90% center crop,
  SigLIP/DINOv2 channel stacking), prompt template + trailing empty token `29871`,
  QUANTILES↔BOUNDS_Q99, and the parallel-decoding mask — any mismatch skews the actions. M5 uses
  `sample_libero_spatial_observation.pkl` to compare the new vs old 8×7 chunk element-wise
  (atol 1e-3); it gates LIBERO eval and must pass first.
- **Version gap (mostly closed)**: both the reference submodule and the installed version are now
  lerobot 0.5.2 (with torch 2.10.0). Still confirm the installed version exposes `LiberoProcessorStep`
  and the 8-dim state behavior.

## Reference assets (`references/`, read-only sources, do not modify)

`references/{lerobot,openvla,openvla-oft}` are git submodules and the source of truth for the
implementation:
- LeRobot examples: `references/lerobot/src/lerobot/policies/{pi0,act,smolvla}/`,
  `processor/{normalize_processor,tokenizer_processor,pipeline,env_processor}.py`.
- OFT model / action head / projectors: upstream truth at
  `references/openvla-oft/prismatic/{extern/hf/modeling_prismatic.py, models/action_heads.py, models/projectors.py}`;
  the working copy is **vendored** at `src/lerobot_policy_openvla_oft/prismatic/` (edit that, not `references/`).
- OFT image/prompt preprocessing reference: `references/openvla-oft/experiments/robot/openvla_utils.py`.
- Per-file code walkthrough (Chinese, for first-time VLA readers): `references/openvla-oft-程式碼解析.md`.

## Conventions

- Code comments in English only; commit / PR titles and bodies in English (Conventional Commits).
- pre-commit is configured (ruff format + lint, pyupgrade py312, prettier markdown, gitleaks, bandit,
  mypy).
- After finishing a practical unit of work, update the status and checkboxes in `progress.md`.
