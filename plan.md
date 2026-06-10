# 計畫：將 OpenVLA-OFT 遷移為 LeRobot-native policy plugin（推論 + LIBERO 評測）

> 配套文件：進度追蹤見 `progress.md`；程式碼細節解析見 `references/openvla-oft-程式碼解析.md`。

## Context（為什麼做這件事）

本 repo `lerobot-policy-openvla-oft` 的目標是把 `references/openvla-oft` 包成 LeRobot 的 out-of-tree
policy plugin `lerobot_policy_openvla_oft`，載入 `moojink/openvla-7b-oft-finetuned-libero-spatial`，
用 `lerobot-eval --env.type=libero` 驗證成功率是否接近論文（LIBERO-Spatial 約 97%）。

目前 `src/lerobot_policy_openvla_oft/` 只有三個 stub 檔（全部 `raise NotImplementedError`），
且尚未被安裝成 distribution。本計畫定義「從現況到可在 LIBERO 跑出成功率」的每一步與先後順序。

決策方向（已確認）：
- **架構：LeRobot-native 重寫**（非 vendor OFT 推論黑箱）。
- **範圍：推論 + LIBERO 評測**（`forward()` 訓練本輪先 `raise NotImplementedError`）。

### 「native 重寫」的務實界定

不是把 7B VLM 從零重寫。界定為：
1. **模型本體**：把 OFT 的 model 類別（`PrismaticVisionBackbone`、`PrismaticProjector`、
   `OpenVLAForActionPrediction` 的平行解碼 forward、`L1RegressionActionHead`、`ProprioProjector`）
   乾淨移植進 plugin 的 `model/` 子模組，**沿用 moojink 預訓練權重**，但不 import `prismatic` 套件
   （避開其 `__init__` 拖進 dlimp / tensorflow 訓練鏈）。
2. **所有 I/O 與正規化**：改用 LeRobot 慣例，即 `NormalizerProcessorStep` / `UnnormalizerProcessorStep`、
   `TokenizerProcessorStep`、自訂影像 `ProcessorStep`，而非 OFT 的 `openvla_utils.get_vla_action` 或
   模型內建的 `_unnormalize_actions`。模型只吐 normalized 動作，反正規化交給 postprocessor。
3. **v1 範圍裁剪**：LIBERO-Spatial 用 `use_l1_regression=True, use_diffusion=False, use_film=False,
   num_images_in_input=2, use_proprio=True, chunk=8, action_dim=7`。
   因此 v1 **不移植 FiLM、不移植 diffusion / noisy action projector**，大幅縮小工作量。

## 查證到的關鍵事實（已讀原始碼確認）

- **Plugin 探索機制**：`references/lerobot/src/lerobot/utils/import_utils.py:213`
  `register_third_party_plugins()` 直接 `importlib.import_module(dist_name)`，所以發行名必須是可 import
  的底線名 `lerobot_policy_openvla_oft`（現有 pyproject 已是底線，OK）。但 **plugin 目前未安裝成
  distribution**（`importlib.metadata` 只看得到 `lerobot`），且 `pyproject.toml` **沒有 `[build-system]`**，
  必須補上才能 editable 安裝並被探索到。
- **命名 bug（必修）**：`references/lerobot/src/lerobot/policies/factory.py:620-632`
  第三方 policy 類別由 `config類別名.removesuffix("Config") + "Policy"` 推導，module 由
  `configuration_` → `modeling_` 推導。現有 config 叫 `OpenvlaOftPolicyConfig` 會推出
  `OpenvlaOftPolicyPolicy`，但 modeling 是 `OpenvlaOftPolicy`，eval 時會 `AttributeError`。
- **processor 推導**：`factory.py:651-660` 由 `config.type`（註冊字串）組 `make_{type}_pre_post_processors`，
  module 由 `configuration_` → `processor_`。
- **修正方案（比照 pi0）**：註冊字串 `openvla_oft`、config 類別 `OpenvlaOftConfig`、
  policy 類別 `OpenvlaOftPolicy`（`name="openvla_oft"`）、工廠 `make_openvla_oft_pre_post_processors`。
  載入用 `--policy.type=openvla_oft`（dev）或 `--policy.path=<hub repo>`（轉檔後）。
- **正規化全在 processor**：`PreTrainedPolicy`（`pretrained.py`）本體不做正規化；
  `NormalizerProcessorStep`/`UnnormalizerProcessorStep`（`processor/normalize_processor.py`）由
  `features + norm_map + dataset_stats` 建構。LeRobot 的 `NormalizationMode.QUANTILES` 用 q01/q99 映射到
  [-1,1]，**等價於 OFT 的 `BOUNDS_Q99`**（LIBERO 動作與 proprio 都用這個），可直接對上。
- **action queue 範式**：`act/modeling_act.py:100-123`、`pi0/modeling_pi0.py:1247-1262`
  `select_action` 在 queue 空時呼叫 `predict_action_chunk`，取前 `n_action_steps`、`transpose(0,1)` 入 deque、
  `popleft()`。直接照抄。
- **LIBERO env 已替我們做好觀測**：`references/lerobot/src/lerobot/processor/env_processor.py`
  `LiberoProcessorStep` 已做影像 180° 翻轉（`torch.flip(img, dims=[2,3])`）與組出 8 維 proprio
  （`eef_pos(3) + quat2axisangle(3) + gripper_qpos(2)` → `observation.state`），與 OFT 的 LIBERO 觀測
  **完全一致，不要重做**。env 動作維度 7、範圍 [-1,1]；影像鍵 `observation.images.image`（agentview）與
  `observation.images.image2`（eye_in_hand）。`--env.task` 可為 `libero_spatial/object/goal/10`。
- **eval 流程**：`scripts/lerobot_eval.py` 的 rollout 順序為
  `env_preprocessor（LiberoProcessorStep）→ policy preprocessor → policy.select_action → policy postprocessor`。
- **版本落差**：reference submodule 是 lerobot 0.5.2，實際安裝是 0.5.1 + torch 2.10.0。M0 需對齊並確認
  installed 版本含 `LiberoProcessorStep` 與 8 維 state（reference 有，installed 待確認）。

## 里程碑與先後順序

依賴關係：M0 →（M1、M2 可並行）→ M3、M4 → M5（正確性 gate）→ M6（LIBERO eval）→ M7、M8（可選）。

### M0 — 封裝與環境調和（讓 plugin 可安裝、可被探索；修命名 bug；釘相依）
- `pyproject.toml`：補 `[build-system]`（hatchling 或 setuptools）+ src layout 套件設定；
  在 plugin 相依加 `transformers`（moojink fork 4.40.1，或經實測可載 Llama-2 `inputs_embeds` +
  自訂 4D attention mask 的版本）、`timm==0.9.x`、`tokenizers`，與 lerobot 隔離（lerobot base 不依賴
  transformers，可釘舊版不破壞 lerobot）。**v1 不需 diffusers**。
- 修 `src/lerobot_policy_openvla_oft/__init__.py`：目前 import 舊 stub 名（`configuration_my_policy`、
  `MyPolicyConfig` 等）已壞，改成 `OpenvlaOftConfig` / `OpenvlaOftPolicy` /
  `make_openvla_oft_pre_post_processors`。
- 對齊 lerobot 版本（建議升到 0.5.2 對上 reference），`uv sync` 後 `uv pip install -e .`。
- **DoD**：`importlib.metadata` 看得到 `lerobot_policy_openvla_oft`；`register_third_party_plugins()` 能
  import 成功；確認 installed lerobot 有 `LiberoProcessorStep`。

### M1 — 移植 OFT 模型最小子集（沿用權重，不靠 prismatic 套件）
- 在 `modeling_openvla_oft.py`（或新增 `model/` 子模組）移植：
  `PrismaticVisionBackbone`（SigLIP+DINOv2 融合、多影像）、`PrismaticProjector`、
  `OpenVLAForActionPrediction` 的平行解碼 forward 與 `predict_action`（**移除內建反正規化**，只回 normalized
  動作）、`L1RegressionActionHead`（`references/openvla-oft/prismatic/models/action_heads.py`）、
  `ProprioProjector`（`projectors.py`）。Llama-2 decoder 用 `transformers` 的 `AutoModelForCausalLM` 載入。
- 權重載入：從 `moojink/openvla-7b-oft-finetuned-libero-spatial` 取「合併後的 base VLA」+
  `action_head--*.pt` + `proprio_projector--*.pt` + `dataset_statistics.json`，灌進移植後的模組。
- **DoD**：能在 CPU 上 `from_pretrained` 並對假輸入跑出 `(1, 8, 7)` 形狀的 normalized 動作，不 import 到
  tensorflow / dlimp。

### M2 — Config 類別 `OpenvlaOftConfig`
- `@PreTrainedConfig.register_subclass("openvla_oft")`，欄位含：`pretrained_oft_repo`（moojink repo 或本地）、
  `chunk_size=8`、`n_action_steps=8`、`n_obs_steps=1`、`num_images_in_input=2`、`use_proprio=True`、
  tokenizer 與影像參數（`image_size=224`、`center_crop=True`）。
- `normalization_mapping = {VISUAL: IDENTITY, STATE: QUANTILES, ACTION: QUANTILES}`
  （影像正規化在自訂影像 step 內做；STATE/ACTION 對應 OFT BOUNDS_Q99）。
- 實作 `validate_features()`（需要 2 影像 + state + action）、
  `action_delta_indices = list(range(chunk_size))`、`observation_delta_indices = None`、
  `reward_delta_indices = None`、`get_optimizer_preset()`（AdamW，雖本輪不訓練仍需提供）、
  `get_scheduler_preset() -> None`。
- 移除現有檔尾多餘的 `raise NotImplementedError`。

### M3 — Processor pipeline `make_openvla_oft_pre_post_processors`
- 參考 `pi0/processor_pi0.py`、`smolvla/processor_smolvla.py` 的 step 順序。
- **preprocessor**：`RenameObservationsProcessorStep`（把 `observation.images.image`→primary、
  `observation.images.image2`→wrist，`observation.state`→proprio）→ `AddBatchDimensionProcessorStep` →
  **自訂 `OpenvlaOftImageProcessorStep`**（resize 224 + center crop 90% + SigLIP/DINOv2 channel stacking，
  複刻 `experiments/robot/openvla_utils.py` 的 `resize_image_for_policy`/`center_crop_image` 與
  `prismatic` 影像 transform，用 `@ProcessorStepRegistry.register` 註冊）→ prompt 模板 +
  `TokenizerProcessorStep`（Llama-2 tokenizer，含 OFT 的 `In: What action should the robot take to {task}?\nOut:`
  與結尾空 token 29871 細節）→ `DeviceProcessorStep` → `NormalizerProcessorStep`（STATE/ACTION QUANTILES，
  用 OFT 統計轉成的 `dataset_stats`）。
- **postprocessor**：`UnnormalizerProcessorStep`（ACTION QUANTILES）→ `DeviceProcessorStep(cpu)`。
- 寫一個小工具把 OFT `dataset_statistics.json`（q01/q99/min/max/mean/std）轉成 LeRobot `dataset_stats`
  dict（鍵用 `observation.state` / `action`）。
- **DoD**：對 sample 觀測，preprocessor 產出的 `pixel_values` / `input_ids` / normalized `proprio` 與原版
  OFT `processor` 的輸出數值吻合（逐項比對）。

### M4 — Policy 類別 `OpenvlaOftPolicy` + 匯出
- `config_class = OpenvlaOftConfig`、`name = "openvla_oft"`。
- `__init__(self, config, **kwargs)`：`super().__init__(config)` → `config.validate_features()` →
  組裝並載入 M1 的模型 → `self.reset()`。
- `reset()`：`self._action_queue = deque(maxlen=config.n_action_steps)`。
- `predict_action_chunk(batch)`：跑平行解碼 forward，回 `(B, 8, 7)` **normalized** 動作。
- `select_action(batch)`：照抄 ACT/pi0 queue 範式。
- `forward(batch)`：本輪 `raise NotImplementedError`（訓練不在範圍）。
- `get_optim_params()`：回 `{"params": self.parameters()}`。
- 更新 `__init__.py` 匯出三者。

### M5 — 正確性 gate（最便宜、最關鍵）
- 用 `references/openvla-oft/experiments/robot/libero/sample_libero_spatial_observation.pkl`：
  分別跑 (a) 原版 OFT `get_vla_action` 與 (b) 新 plugin 路徑（preprocessor → policy → postprocessor），
  逐元素比對 8×7 動作 chunk（設容差，如 atol 1e-3）。
- 放進 `tests/`（pytest）。**這是進 M6 前的把關**：native 重寫最大風險就是數值不一致
  （影像前處理、prompt/空 token、QUANTILES vs BOUNDS_Q99、平行解碼 mask）。
- **DoD**：兩路徑動作 chunk 數值吻合。

### M6 — LIBERO 端到端評測
- dev 路徑：`lerobot-eval --policy.type=openvla_oft --policy.pretrained_oft_repo=moojink/... \
  --env.type=libero --env.task=libero_spatial --eval.n_episodes=50`（實際旗標以 config 欄位為準）。
- 確認 rollout 串接 `LiberoProcessorStep` → plugin preprocessor → `select_action` → postprocessor 正常。
- **DoD**：跑完 ≥50 episodes，成功率與論文（LIBERO-Spatial 約 97%）同量級；落差大就回 M5 比對中間張量。

### M7 —（可選）轉成 LeRobot 格式 checkpoint 並上傳
- 寫 `convert` 腳本，用 `PreTrainedPolicy.push_model_to_hub` 產出 `config.json` + `model.safetensors` +
  存好內嵌統計的 `policy_preprocessor/postprocessor`，讓 `--policy.path=<repo>` 直接可用、可分享。
- 在 policy MDX 記錄可重現的 `lerobot-eval` 指令與成功率表。

### M8 —（盡力）Mac MPS 相容
- 模型 attention 用 `sdpa`/`eager`（不依賴 flash-attn）、device 由 `config.device` 串接、處理 bf16/fp32。
- 正確性 gate 可在 CPU/MPS 跑；**完整 LIBERO eval 需模擬器且實務上需 CUDA**（論文用 A100），
  大規模評測建議用 GPU box（RunPod）。

## 要動到的關鍵檔案
- `pyproject.toml`（build-system、相依、版本）
- `src/lerobot_policy_openvla_oft/__init__.py`（修壞掉的匯出）
- `src/lerobot_policy_openvla_oft/configuration_openvla_oft.py`（改名 `OpenvlaOftConfig` + 重寫）
- `src/lerobot_policy_openvla_oft/modeling_openvla_oft.py`（重寫；移植 OFT 模型子集，建議拆 `model/`）
- `src/lerobot_policy_openvla_oft/processor_openvla_oft.py`（重寫 + 自訂影像/prompt step + 統計轉換）
- `tests/`（M5 正確性 gate）

## 可重用資產（沿用，勿重寫）
- OFT 模型/動作頭/投影器：`references/openvla-oft/prismatic/{extern/hf/modeling_prismatic.py,
  models/action_heads.py, models/projectors.py}`
- OFT 影像/prompt 前處理參考：`references/openvla-oft/experiments/robot/openvla_utils.py`
- LeRobot 範例：`references/lerobot/src/lerobot/policies/{pi0,act,smolvla}/`、
  `processor/{normalize_processor,tokenizer_processor,pipeline}.py`、
  `processor/env_processor.py`（`LiberoProcessorStep`，**直接靠它，不重做翻轉/組 state**）
- 細節解析：本 repo 的 `references/openvla-oft-程式碼解析.md`

## 主要風險
1. **transformers 版本相容**：移植的 Llama-2 平行解碼路徑（`inputs_embeds` + 自訂 4D mask + legacy
   past_key_values）對 transformers 版本敏感；stock 5.x 不相容。需釘可用版本並在 M1 實測。
2. **數值一致性**：影像前處理（lanczos resize、center crop 90%、SigLIP/DINOv2 channel stacking）、
   prompt 與結尾空 token 29871、QUANTILES↔BOUNDS_Q99 等任一不一致都會讓動作偏掉。M5 gate 專門擋這個。
3. **GPU/模擬器可用性**：完整 LIBERO eval 需 LIBERO 模擬器 + 實務上需 CUDA；Mac 端先做 gate 與小樣本。
4. **lerobot 版本落差**：installed 0.5.1 vs reference 0.5.2，需確認 LIBERO env 行為一致。

## 驗證方式（端到端）
1. M0：`register_third_party_plugins()` 能探索到 plugin；`--policy.type=openvla_oft` 不再 `AttributeError`。
2. M5：pytest 比對 sample pkl 的 8×7 動作 chunk 與原版 OFT 吻合（atol 1e-3）。
3. M6：`lerobot-eval ... --env.type=libero --env.task=libero_spatial --eval.n_episodes=50` 跑出成功率，
   與論文同量級。
4. M8：在 MPS/CPU 上能完成 M5 gate；完整 eval 於 GPU 環境執行。
