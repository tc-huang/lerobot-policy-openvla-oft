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
1. **模型本體（兩階段，已調整）**：OFT 推論需要的 model 類別（`PrismaticVisionBackbone`、`PrismaticProjector`、
   `OpenVLAForActionPrediction` 的平行解碼 forward、`L1RegressionActionHead`、`ProprioProjector`）原始碼，
   已整包 vendored 進 `src/lerobot_policy_openvla_oft/prismatic/`（完整上游套件，66 個 .py）。
   - **Phase A（先能動，本輪 M1）**：直接沿用這份 vendored `prismatic/`，**砍掉推論用不到的檔案與分支**
     （RLDS/OXE 資料管線、FSDP/DDP 訓練、原生載入路徑、FiLM、diffusion），修掉絕對 import 與 import 副作用，
     讓推論路徑可離線載入、不拖 dlimp / tensorflow / diffusers。
   - **Phase B（後重構，非阻塞）**：通過 M5 正確性 gate 後，再一步步把保留檔搬進乾淨的 `model/` 子模組、
     砍死分支，收斂成 plugin-native 結構。
   兩階段都**沿用 moojink 預訓練權重**，且不 import 上游真正的 `prismatic` 套件根（避開其拖進 dlimp / tensorflow）。
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

### M1 — 讓 vendored `prismatic/` 的推論路徑可用（Phase A：先能動）

> 背景：完整上游 `prismatic` 套件已整包 vendored 進 `src/lerobot_policy_openvla_oft/prismatic/`
> （66 個 .py，含推論關鍵的 `extern/hf/modeling_prismatic.py`）。本里程碑**不從零重寫**，而是
> 「就地沿用、砍到能跑」：刪掉推論用不到的檔案與分支、修掉會擋 import 的問題，讓
> `modeling_openvla_oft.py` 能 import 到 `OpenVLAForActionPrediction` / `L1RegressionActionHead` /
> `ProprioProjector` 並對假輸入吐出 `(1, 8, 7)` normalized 動作。乾淨化（搬進 `model/`、砍死分支）
> 留到 **Phase B**（M1 之後、可在 M5 gate 通過後做，不阻塞 M6）。

已查證的 vendored 現況（讀原始碼確認）：
- `OpenVLAForActionPrediction` 在 `prismatic/extern/hf/modeling_prismatic.py:720`；`L1RegressionActionHead` /
  `MLPResNet` 在 `models/action_heads.py:84` / `:59`；`ProprioProjector` 在 `models/projectors.py`。
- **阻塞 1（絕對 import）**：66 檔中 36 檔用 `from prismatic.…` 絕對 import。改名成
  `lerobot_policy_openvla_oft.prismatic` 後 top-level `prismatic` 不存在，`modeling_prismatic.py:24,28` 會
  `ModuleNotFoundError`。
- **阻塞 2（diffusers）**：`action_heads.py:7` module 頂層 `from diffusers… import DDIMScheduler`，與要用的
  `L1RegressionActionHead` 同檔；`modeling_prismatic.py` 本身不 import diffusers（diffusion 分支只在 runtime 走）。
- **阻塞 3（constants 副作用）**：`vla/constants.py:49-86` 在 import 時用 `sys.argv` 偵測平台並 print 6 行。
- **阻塞 4（相依未裝）**：env 目前只有 `lerobot 0.5.2`，`transformers/timm/tokenizers` 全 missing（屬 M0）。
- **好消息**：`prismatic/__init__.py`、`extern/__init__.py`、`extern/hf/__init__.py` 已被清成空檔，eager
  import chain 已斷；`tensorflow/dlimp` 只在 `vla/datasets/rlds/*`，推論路徑不碰。

**M1.A0 砍掉推論用不到的子樹（減面積、移除 tf/dlimp/FiLM/diffusion 來源）**
- 刪：`vla/datasets/`（整套 RLDS/OXE 訓練資料管線，唯一會拖 tensorflow/dlimp 的地方）、
  `training/strategies/`、`training/{materialize,metrics}.py`（FSDP/DDP/metrics，訓練用）、
  `models/{vlms,vlas,backbones}/`、`models/{load,materialize,registry}.py`、`models/film_vit_wrapper.py`、
  `vla/{materialize,action_tokenizer}.py`（原生 prismatic 載入路徑與 FiLM，HF extern 推論路徑用不到）。
- 保留：`extern/hf/{modeling_prismatic,configuration_prismatic,processing_prismatic}.py`、
  `models/{action_heads,projectors}.py`、`vla/constants.py`、`training/train_utils.py`（兩個 action mask）。
- 砍完逐一確認保留檔的 import 不再指向被刪檔（grep `from prismatic` 收斂到保留集合內）。

**M1.A1 修絕對 import**
- 把保留檔內 `from prismatic.…` 改成套件絕對 `from lerobot_policy_openvla_oft.prismatic.…`（或相對）。範圍只剩
  少數：`modeling_prismatic.py`（train_utils、constants）、`action_heads.py`（constants）、`train_utils.py` 等。

**M1.A2 去 diffusers（只留 L1 路徑）**
- `action_heads.py`：移除 `from diffusers…` import，刪 `SinusoidalPositionalEncoding` / `NoisePredictionModel` /
  `DiffusionActionHead`，只保留 `MLPResNet` + `L1RegressionActionHead`。
- `modeling_prismatic.py` 的 diffusion 分支（`predict_action` 內 `use_diffusion` 區段、約 `:798-870`）import-time
  不碰 diffusers，**Phase A 先留著不動**，Phase B 再砍死分支。

**M1.A3 去 constants.py 的 import 副作用**
- `vla/constants.py`：把 `sys.argv` 平台偵測與 6 行 print 換成直接寫死 LIBERO 常數
  （`NUM_ACTIONS_CHUNK=8`、`ACTION_DIM=7`、`PROPRIO_DIM=8`、`ACTION_PROPRIO_NORMALIZATION_TYPE=BOUNDS_Q99`、
  `IGNORE_INDEX=-100`、`ACTION_TOKEN_BEGIN_IDX=31743`、`STOP_INDEX=2`），保留同名 export 不變。

**M1.A4 權重載入 + 對外介面（包裝 `OpenvlaOftModel`）**
- 比照 `references/openvla-oft/experiments/robot/openvla_utils.py`，`__init__` 依序載入：base VLA
  （`OpenVLAConfig.from_pretrained` → `OpenVLAForActionPrediction.from_pretrained(..., low_cpu_mem_usage=True)`，
  **不 `trust_remote_code`**，用 vendored 類別；載入後 `vision_backbone.set_num_images_in_input(2)`）、action head
  （`hf_hub_download` checkpoint → 去 DDP `module.` 前綴 → `L1RegressionActionHead.load_state_dict`）、proprio
  projector（同上 → `ProprioProjector(llm_dim, 8)`）、`dataset_statistics.json`（先存，M3 才轉 `dataset_stats`）。
- 對外 `predict_action_chunk(pixel_values, input_ids, attention_mask, proprio)`：呼叫
  `base_vla.predict_action(..., action_head=…, proprio=…, proprio_projector=…, use_film=False,
  noisy_action_projector=None)`，把 `(8,7)` → `(1,8,7)` normalized（batch=1 假設，**不在模型內反正規化**）。
- checkpoint 步數先寫死 spatial（150000）；其餘 repo 之後再做對照表。

- **DoD（Phase A）**：
  1. `import lerobot_policy_openvla_oft.prismatic.extern.hf.modeling_prismatic` 與後續載入不報錯，且
     `sys.modules` 不含 `tensorflow` / `dlimp` / `diffusers`。
  2. CPU 上 `OpenvlaOftModel(repo="moojink/openvla-7b-oft-finetuned-libero-spatial")` 載入 base VLA + action head
     + proprio projector 成功。
  3. 假輸入（`pixel_values (1,12,224,224)`、合法 `input_ids`/`attention_mask`、`proprio (1,8)`）跑
     `predict_action_chunk` 回 `(1, 8, 7)` 無例外。
  4. 釘定可用 transformers 版本（先試 fork `4.40.1`；衝突則記錄實測可載 Llama-2 `inputs_embeds` 路徑的版本）。

---

#### Phase B 漸進重構參考（M5 gate 通過後再做，非阻塞）

> 目標：把上面保留的 vendored 檔逐步搬進乾淨的 `model/` 子模組、移除 `modeling_prismatic.py` 的 diffusion/FiLM
> 死分支與推論未用檔（如 `processing_prismatic.py`），收斂成 plugin-native。以下 M1.0–M1.7 為原規畫的「從零抽
> `model/`」細目，作為 Phase B 的重構藍圖；**來源改為已 vendored 的 `prismatic/`**（不再從 `references/` 複製）。

**M1.0 建子模組骨架 + 內聯常數（切斷 prismatic 依賴鏈）**
- 新增 `src/lerobot_policy_openvla_oft/model/`，內含：
  - `constants.py`：直接寫死 LIBERO 常數（`NUM_ACTIONS_CHUNK=8`、`ACTION_DIM=7`、`PROPRIO_DIM=8`、
    `IGNORE_INDEX=-100`、`ACTION_TOKEN_BEGIN_IDX=31743`、`STOP_INDEX=2`）。**不要** import
    `prismatic.vla.constants`（那支會在 import 時 print 並依 `sys.argv` 自動選平台）。
  - `masking.py`：複製 `get_current_action_mask` / `get_next_actions_mask`
    （`references/openvla-oft/prismatic/training/train_utils.py:8-39`），只依賴上面的 `constants`。

**M1.1 移植 HF 內部模型設定（注意：不是 M2 的 policy config）**
- 把 `references/openvla-oft/prismatic/extern/hf/configuration_prismatic.py` 的 `PrismaticConfig` +
  `OpenVLAConfig` 原樣複製進 `model/configuration_prismatic.py`（只依賴 `transformers`，無 prismatic 依賴）。
- moojink-libero-spatial 的 `config.json` 會帶 `vision_backbone_id="dinosiglip-vit-so-224px"`、
  `llm_backbone_id="llama2-7b-pure"`、`use_fused_vision_backbone=True`、`text_config`(LlamaConfig)、
  `norm_stats`、`n_action_bins=256` 等，**由 hub config.json 還原即可，不用自己填**。
- 這個 config 是給 HF `from_pretrained` 當「權重容器設定」用，**與 M2 的 `OpenvlaOftConfig`（policy 設定）是兩回事**。

**M1.2 移植視覺 backbone + projector（純 nn.Module）**
- `PrismaticVisionBackbone`（`modeling_prismatic.py:67-227`）整段照搬，含：`_create_featurizer`
  （`timm.create_model(pretrained=False)` + monkey-patch `forward` 取倒數第二層
  `get_intermediate_layers`）、`_patch_layer_scales`（把 LayerScale 的 `gamma` 改名 `scale_factor`，
  避開 HF 對含 `gamma` 參數的覆寫）、多影像 `forward`（每張影像 6 channel：SigLIP 前 3、DINOv2 後 3）。
- `PrismaticProjector`（`modeling_prismatic.py:231-262`）整段照搬（fused 版 fc1/fc2/fc3 + 2 個 GELU）。
- 依賴只剩 `timm`（**0.9.10/0.9.11/0.9.12/0.9.16**，modeling 內有版本斷言）、`torch`。

**M1.3 移植主模型骨幹（砍掉 diffusion / FiLM / discrete 分支）**
- 複製 `PrismaticPreTrainedModel` + `PrismaticForConditionalGeneration`，保留：`__init__`
  （vision_backbone + projector + `AutoModelForCausalLM.from_config(text_config)` 載 Llama-2 decoder）、
  HF boilerplate（get/set embeddings…）、`_process_action_masks`、`_process_vision_features`
  （**只留無 FiLM 分支**）、`_process_proprio_features`、`_build_multimodal_attention`、
  `_build_multimodal_labels`、multimodal `forward`。
- **砍掉**：`_replace_input_embeddings`、`noisy_actions` / `noisy_action_projector` /
  `diffusion_timestep_embeddings` 相關分支、所有 FiLM 路徑（縮小面積、避免拖 diffusers / peft）。

**M1.4 移植 `predict_action`（只留 L1 regression，且移除內建反正規化）**
- 保留 `_prepare_input_for_action_prediction`（補 `ACTION_DIM*NUM_ACTIONS_CHUNK=56` 個 placeholder
  action token + 1 個 stop token，並延長 attention_mask）、`_prepare_labels_for_action_prediction`、
  `_regression_or_discrete_prediction`（**只留 `action_head is not None` 的 L1 分支**）。
- 保留 predict_action 開頭「若結尾不是 token `29871` 就補空 token」的邏輯（與訓練輸入對齊），以及
  `NUM_PATCHES = get_num_patches() * num_images_in_input`、`use_proprio` 時 `+1` 的計算。
- **關鍵改動**：predict_action **回傳 normalized 動作（reshape 成 `(8, 7)`），不呼叫 `_unnormalize_actions`**；
  反正規化交給 M3 的 `UnnormalizerProcessorStep`。`norm_stats` / `get_action_stats` /
  `_unnormalize_actions` / discrete bins（`self.bins` / `bin_centers`）推論路徑都不會用到（可不移植或保留不接）。
- 注意：這條路徑是「一次 forward 吃 `inputs_embeds` + 2D attention_mask，由 Llama 內部自建 causal mask」，
  **沒有任何自訂 4D mask**；所謂「平行解碼」指一次插入全部 action placeholder token、再用 L1 head 讀
  action 位置的 hidden states，不是 bidirectional 改寫。版本敏感點在 `inputs_embeds` / `position_ids=None` /
  legacy `past_key_values` 的處理（見「主要風險 1」）。

**M1.5 移植 action head + proprio projector**
- `L1RegressionActionHead`（`action_heads.py:84-107`，含 `MLPResNet` / `MLPResNetBlock`）照搬；
  其 `input_dim=llm_dim*ACTION_DIM`、輸出 reshape 成 `NUM_ACTIONS_CHUNK` 都改 import M1.0 的 `constants`。
- `ProprioProjector(llm_dim, proprio_dim=8)`（`projectors.py:6-24`）照搬。
- **砍掉**：`SinusoidalPositionalEncoding` / `NoisePredictionModel` / `DiffusionActionHead`（會拖
  `diffusers`）與 `NoisyActionProjector`。

**M1.6 權重載入（沿用 moojink；比照 `experiments/robot/openvla_utils.py`）**
- 設一個包裝類別 `OpenvlaOftModel`，`__init__` 依序載入：
  1. **base VLA**：先 `OpenVLAConfig.from_pretrained(repo)`，再
     `OpenVLAForActionPrediction.from_pretrained(repo, config=cfg, torch_dtype=..., low_cpu_mem_usage=True)`，
     **不要 `trust_remote_code=True`**（避免拉 hub 上那份會 import prismatic 的 modeling 檔），用移植版類別；
     載入後 `vision_backbone.set_num_images_in_input(2)`。
  2. **action head**：`hf_hub_download(repo, "action_head--150000_checkpoint.pt")` →
     `load_component_state_dict`（去掉 DDP `module.` 前綴，見 `openvla_utils.py:230-250`）→
     `L1RegressionActionHead(...).load_state_dict(...)`。
  3. **proprio projector**：`hf_hub_download(repo, "proprio_projector--150000_checkpoint.pt")` → 同上 →
     `ProprioProjector(llm_dim, 8).load_state_dict(...)`。
  4. **dataset_statistics.json**：`hf_hub_download(repo, "dataset_statistics.json")` 先存下（**M3 才會**轉成
     LeRobot `dataset_stats`；M1 因為不在模型內反正規化，不需用到）。
- checkpoint 步數隨 repo 不同（spatial/object/10 = 150000、goal = 50000、混合 = 300000，見
  `openvla_utils.py:497-503`）；M1 先寫死 spatial，或做 `repo → 檔名` 小對照表，不必通用化。

**M1.7 對外推論介面**
- 在 `OpenvlaOftModel` 上提供 `predict_action_chunk(pixel_values, input_ids, attention_mask, proprio)`，
  內部呼叫 `base_vla.predict_action(..., action_head=self.action_head, proprio=proprio,
  proprio_projector=self.proprio_projector, use_film=False, noisy_action_projector=None)`，把回傳的 `(8,7)`
  轉成 torch `(1, 8, 7)` normalized 動作。**M4 的 policy 會呼叫這個**。
- 原版 `predict_action` 回傳 numpy 且 batch 寫死 1（LIBERO 單環境逐步呼叫）；M1 沿用 batch=1 假設，
  `(8,7)` → `unsqueeze(0)` → `(1,8,7)`。device/dtype 在 M1.7 統一轉好再傳（predict_action 內部會
  `torch.Tensor(proprio)` 重建 tensor，留意 device/dtype）。

- **DoD（Phase B 重構後須維持）**：
  1. CPU 上 `OpenvlaOftModel(repo="moojink/openvla-7b-oft-finetuned-libero-spatial")` 能成功載入
     base VLA + action head + proprio projector，過程 `sys.modules` 不含 `tensorflow` / `dlimp` / `diffusers`。
  2. 對假輸入（`pixel_values` shape `(1, 12, 224, 224)` ＝ 2 影像 × 6 channel、合法 `input_ids` /
     `attention_mask`、`proprio` shape `(1, 8)`）跑 `predict_action_chunk`，回傳 `(1, 8, 7)` normalized 動作、無例外。
  3. 釘定可用 transformers 版本（先試 moojink fork `4.40.1`；若與 lerobot 安裝樹衝突，記錄實測可載
     Llama-2 `inputs_embeds` 路徑的最低版本）。

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
1. **transformers 版本相容**：移植的 Llama-2 平行解碼路徑吃 `inputs_embeds` + 2D attention_mask
   （由 Llama 內部自建 causal mask，**無自訂 4D mask**），對 `inputs_embeds` / `position_ids=None` /
   legacy `past_key_values` 的處理在不同 transformers 版本下行為不同；stock 5.x 不相容。需釘可用版本並在 M1 實測。
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
