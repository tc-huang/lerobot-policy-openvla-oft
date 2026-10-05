# lerobot-policy-openvla-oft

[English](README.md) | 繁體中文

本專案是 [LeRobot](https://github.com/huggingface/lerobot) 的 out-of-tree
policy 插件，將 [OpenVLA-OFT](https://openvla-oft.github.io/)（Kim、Finn 與
Liang，2025）移植到 LeRobot v0.6.1，以 `--policy.type openvla_oft` 使用。插件
可將原作者釋出的官方 LIBERO checkpoint 轉換為 LeRobot 格式，並以
`lerobot-eval` 評估，目標是復現論文中的 LIBERO 結果。此外也支援以
`lerobot-train` 從 `openvla/openvla-7b` 進行 LoRA fine-tune，並以
`lerobot-rollout` 將 fine-tune 後的 policy 部署到單臂 SO-100 或 SO-101
follower 手臂上。

## 設計

本專案依照論文逐一建構 policy 的各個元件。以下每一節說明一個元件，以及它與論文和
原始實作的對應關係。

來源標示方式如下：

- **論文**：Kim、Finn 與 Liang，
  [Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success](https://arxiv.org/abs/2502.19645)，
  arXiv:2502.19645v2。
- **原 repo**：[moojink/openvla-oft](https://github.com/moojink/openvla-oft)
  的 `e4287e9`，以 submodule 放在 `third_party/openvla-oft`。路徑皆相對於該目錄。
- **本專案**：本專案自行決定，並非取自上述任一來源。

### 1. Configuration

`OpenVLAOFTConfig` 註冊 policy type `openvla_oft`，並定義輸入與輸出的規格、正規化
方式，以及訓練的預設設定。預設值沿用 LIBERO 的設定。

| 設定                        | 預設值                     | Config 欄位                                       | 來源                                                                                     |
| --------------------------- | -------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Action chunk 大小           | 8                          | `chunk_size`                                      | 論文 §V-A、Table IV；原 repo `prismatic/vla/constants.py:27`                             |
| 每個 chunk 執行的 action 數 | 8（整個 chunk，open-loop） | `n_action_steps`                                  | 論文 §V-A、Table IV；原 repo `experiments/robot/libero/run_libero_eval.py:100`           |
| 觀測歷史                    | 無（只用當下這一步）       | `n_obs_steps`、`observation_delta_indices`        | 論文 Table IV                                                                            |
| State 與 action 正規化      | `[q01, q99]` → `[-1, 1]`   | `normalization_mapping`（`QUANTILES`）            | 原 repo `prismatic/vla/constants.py:30`；論文只提到 action 正規化到 `[-1, 1]`（App. D）  |
| 影像正規化                  | 無                         | `normalization_mapping`（`IDENTITY`）             | 本專案：每個 vision backbone 會自行正規化                                                |
| Optimizer                   | AdamW                      | `get_optimizer_preset()`                          | 原 repo `vla-scripts/finetune.py:935`；論文未提及                                        |
| Learning rate               | 5e-4                       | `optimizer_lr`                                    | 論文 Table IV；原 repo `vla-scripts/finetune.py:89`                                      |
| Weight decay                | 0.01                       | `optimizer_weight_decay`                          | 原 repo：`vla-scripts/finetune.py:935` 未設定，因此是 PyTorch AdamW 的預設值；論文未提及 |
| Gradient clipping           | 無                         | `optimizer_grad_clip_norm`（0）                   | 原 repo：`vla-scripts/finetune.py` 沒有做 clipping；論文未提及                           |
| Learning rate 衰減          | 100K 步後 ×0.1             | `scheduler_decay_steps`、`scheduler_decay_factor` | 論文 App. D、Table IV；原 repo `vla-scripts/finetune.py:91`、`:941-944`                  |
| Learning rate warmup        | 無                         | （不支援）                                        | 原 repo `vla-scripts/finetune.py:90`；論文未提及                                         |

原始實作在 import 時透過檢查命令列參數來決定 chunk 大小與正規化方式
（`prismatic/vla/constants.py`）；本專案則將它們明確定義為 configuration 欄位。

LeRobot 的 `QUANTILES` 與原始實作的 `BOUNDS_Q99` 有兩處不同：原始實作會把正規化
後的值截斷到 `[-1, 1]`（`prismatic/vla/datasets/rlds/utils/data_utils.py:81`），
並且不正規化被 mask 的維度。例如 LIBERO checkpoint 的 `dataset_statistics.json`
把 gripper 的 action 維度設為 mask。這兩處差異都由 processor 處理。

論文 Table IV 中其餘的超參數，由其他元件或訓練指令負責：

| 設定（論文 Table IV） | 值                            | 負責的元件                                |
| --------------------- | ----------------------------- | ----------------------------------------- |
| 總 batch size         | 64（每張 GPU 8 × 8 張 GPU）   | `lerobot-train --batch_size`、多 GPU 訓練 |
| 訓練步數              | 150K（LIBERO-Goal 為 50K）    | `lerobot-train --steps`                   |
| 輸入影像              | 1 張第三人稱 + 1 張手腕相機   | Dataset features；§2 Vision backbone      |
| 機器人狀態輸入        | 是                            | Dataset features；§3 Projectors           |
| 輸入影像尺寸          | 224 × 224                     | §2 Vision backbone（`image_size`）        |
| LoRA rank             | 32                            | LoRA fine-tuning                          |
| 影像增強              | 90% random crop、color jitter | Processor 與訓練                          |
| FiLM                  | 否                            | 不在範圍內                                |

### 2. Vision backbone

`FusedVisionBackbone`（`vision_backbone.py`）是 OpenVLA 的融合式 vision
encoder。每張相機影像都會同時經過 DINOv2 與 SigLIP 兩個 vision transformer（以
`timm` 建立）。兩者各自產生 256 個 patch feature，並沿 channel 維度串接
（1024 + 1152 = 2176）。所有相機影像的 feature 再沿序列維度串接，因此兩台相機會
產生 512 個 token。

| 設定                        | 值                                                                                                                              | Config 欄位                  | 來源                                                                                                                                                                                           |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Vision transformer          | DINOv2 ViT-L/14（含 4 個 register）、SigLIP SO400M/14                                                                           | （固定）                     | 論文 App. A；原 repo `prismatic/extern/hf/configuration_prismatic.py:36`                                                                                                                       |
| 輸入影像尺寸                | 224 × 224                                                                                                                       | `image_size`                 | 論文 Table IV；原 repo `prismatic/extern/hf/configuration_prismatic.py:22`                                                                                                                     |
| 每張影像的 patch feature 數 | 每個 transformer 256 個                                                                                                         | （推導而得）                 | 論文 App. A                                                                                                                                                                                    |
| 取用的 feature 層           | 倒數第二個 block 的輸出，不經過最後的 norm                                                                                      | （固定）                     | 原 repo `prismatic/extern/hf/modeling_prismatic.py:137`；論文未提及                                                                                                                            |
| Prefix token                | 捨棄（DINOv2 的 CLS 與 register token）                                                                                         | （固定）                     | 原 repo：`modeling_prismatic.py:137` 中 `get_intermediate_layers` 的預設行為；論文未提及                                                                                                       |
| 融合方式                    | DINOv2 與 SigLIP 的 feature 沿 channel 串接                                                                                     | （固定）                     | 論文 App. A；原 repo `prismatic/extern/hf/modeling_prismatic.py:223`                                                                                                                           |
| 多張影像                    | 共用 backbone，沿序列串接                                                                                                       | （由 dataset features 決定） | 論文 App. A（OFT 修改 1）；原 repo `prismatic/extern/hf/modeling_prismatic.py:210-227`                                                                                                         |
| 像素正規化                  | DINOv2：mean `(0.484375, 0.455078125, 0.40625)`、std `(0.228515625, 0.2236328125, 0.224609375)`；SigLIP：mean 與 std 皆為 `0.5` | （固定）                     | 原 repo：釋出的 `preprocessor_config.json` 中的 `tvf_normalize_params`，由 `prismatic/extern/hf/processing_prismatic.py:139` 套用，訓練（`vla-scripts/finetune.py:973`）與推論皆同；論文未提及 |

DINOv2 的統計值是 ImageNet 的 mean 與 std 經 bfloat16 捨入後的結果。這個捨入被寫
進了釋出的 processor，因此 checkpoint 正是以這些數值訓練的。

與原始實作的差異（本專案）：

- **輸入格式。** 原始實作把每張影像放兩次，疊成 `(B, 6 × 影像數, H, W)` 的
  tensor，並由 processor 分別依兩個 transformer 的需求正規化。本專案的 backbone
  接收值域為 `[0, 1]` 的 `(B, 影像數, 3, H, W)`，並自行套用各 transformer 的正規
  化，讓 processor 不必知道模型細節。
- **移除不會用到的層。** 原始實作建立完整的 transformer，但只讀取倒數第二個
  block。本專案在建立模型時就移除最後一個 block、最後的 norm，以及 SigLIP 的
  attention-pooling head。這些參數不影響輸出，因此不會收到 gradient；原始實作是
  用 `DistributedDataParallel(find_unused_parameters=True)` 繞過這個問題
  （`vla-scripts/finetune.py:875`）。移除這些參數可以省去多 GPU 訓練時的額外負
  擔。Checkpoint 中對應的權重會在轉換時捨棄。
- **LayerScale 參數名稱。** 原始實作把 timm LayerScale 的 `gamma` 參數改名為
  `scale_factor`，因為 Hugging Face `transformers` 會改寫名稱中含有 `gamma` 的參
  數（`modeling_prismatic.py:141-157`）。本專案不透過 `transformers` 載入權重，因
  此保留 timm 的名稱，改在轉換時重新命名 checkpoint 的 key。
- **批次編碼。** 所有相機影像在一次批次呼叫中編碼，而不是用 Python 迴圈逐張處理。

### 3. Projectors

`projectors.py` 包含兩個 MLP，負責把非文字的輸入映射到 Llama-2 的 token embedding
空間（`llm_dim = 4096`）。

| 設定                        | 值                                                     | Config 欄位                  | 來源                                                                               |
| --------------------------- | ------------------------------------------------------ | ---------------------------- | ---------------------------------------------------------------------------------- |
| Vision projector            | 3 層 GELU MLP：`2176 → 8704 → 4096 → 4096`（71M 參數） | （固定）                     | 論文 App. A、App. B.3；原 repo `prismatic/extern/hf/modeling_prismatic.py:243-246` |
| Vision projector 隱藏層寬度 | vision feature 維度的 4 倍                             | （固定）                     | 原 repo `prismatic/extern/hf/modeling_prismatic.py:243`；論文未提及                |
| Proprio projector           | 2 層 GELU MLP：`state 維度 → 4096 → 4096`              | （固定）                     | 論文 App. A（OFT 修改 2）、App. B.3；原 repo `prismatic/models/projectors.py:6-23` |
| Proprio projector 大小      | LIBERO 8 維 state 時為 17M 參數                        | （推導而得）                 | 論文 Table IV                                                                      |
| Proprio token 數            | 每一步 1 個 token                                      | （固定）                     | 原 repo `prismatic/extern/hf/modeling_prismatic.py:449-459`；論文未提及            |
| State 維度                  | 取自 dataset 的 `observation.state` feature            | （由 dataset features 決定） | 原 repo 針對 LIBERO 寫死 `PROPRIO_DIM = 8`（`prismatic/vla/constants.py:29`）      |

Vision projector 屬於預訓練的 OpenVLA 模型；proprio projector 則是 OpenVLA-OFT
新增的模組，以 PyTorch 的預設方式初始化（`vla-scripts/finetune.py:878-884`）。兩者
各自如何訓練，留待 LoRA fine-tuning 一節說明。

與原始實作的差異（本專案）：

- **Vision projector 只保留一種用途。** 原始的 `PrismaticProjector` 也支援給單一
  （非融合）vision backbone 用的 2 層版本。OpenVLA 一律使用融合 backbone，因此只
  保留 3 層版本。
- **Proprio token 的 shape。** `ProprioProjector` 直接回傳 `(B, 1, llm_dim)`，也就
  是可以直接插入序列的 token，而不是把 reshape 留給呼叫端處理。
