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

| 設定                           | 預設值                              | Config 欄位                                       | 來源                                                                                                   |
| ------------------------------ | ----------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Action chunk 大小              | 8                                   | `chunk_size`                                      | 論文 §V-A、Table IV；原 repo `prismatic/vla/constants.py:27`                                           |
| 每個 chunk 執行的 action 數    | 8（整個 chunk，open-loop）          | `n_action_steps`                                  | 論文 §V-A、Table IV；原 repo `experiments/robot/libero/run_libero_eval.py:100`                         |
| 觀測歷史                       | 無（只用當下這一步）                | `n_obs_steps`、`observation_delta_indices`        | 論文 Table IV                                                                                          |
| State 與 action 正規化         | `[q01, q99]` → `[-1, 1]`            | `normalization_mapping`（`QUANTILES`）            | 原 repo `prismatic/vla/constants.py:30`；論文只提到 action 正規化到 `[-1, 1]`（App. D）                |
| 不正規化的 action 維度         | 無（每個維度都正規化）              | `action_norm_mask`                                | 原 repo `prismatic/vla/datasets/rlds/oxe/materialize.py:35-45`；見 §8.2                                |
| 影像正規化                     | 無                                  | `normalization_mapping`（`IDENTITY`）             | 本專案：每個 vision backbone 會自行正規化                                                              |
| Optimizer                      | AdamW                               | `get_optimizer_preset()`                          | 原 repo `vla-scripts/finetune.py:935`；論文未提及                                                      |
| Learning rate                  | 5e-4                                | `optimizer_lr`                                    | 論文 Table IV；原 repo `vla-scripts/finetune.py:89`                                                    |
| Weight decay                   | 0.01                                | `optimizer_weight_decay`                          | 原 repo：`vla-scripts/finetune.py:935` 未設定，因此是 PyTorch AdamW 的預設值；論文未提及               |
| Gradient clipping              | 無                                  | `optimizer_grad_clip_norm`（0）                   | 原 repo：`vla-scripts/finetune.py` 沒有做 clipping；論文未提及                                         |
| Learning rate 衰減             | 100K 步後 ×0.1                      | `scheduler_decay_steps`、`scheduler_decay_factor` | 論文 App. D、Table IV；原 repo `vla-scripts/finetune.py:91`、`:941-944`                                |
| Learning rate warmup           | 無                                  | （不支援）                                        | 原 repo `vla-scripts/finetune.py:90`；論文未提及                                                       |
| 權重 dtype                     | bfloat16                            | `dtype`                                           | 原 repo `vla-scripts/finetune.py:837`（模型）、`:895`（action head）；論文未提及                       |
| Proprio projector 權重         | float32                             | `proprio_projector_fp32`                          | 原 repo `vla-scripts/finetune.py:878-884`（未設定 `to_bf16`）；論文未提及                              |
| 混合精度                       | Forward 在 bfloat16 autocast 下執行 | （跟隨 `dtype`）                                  | 原 repo `vla-scripts/finetune.py:327`；論文未提及                                                      |
| 補值的 chunk 步驟是否計入 loss | 計入，其值為最後一個 action 的複本  | `mask_padded_actions`（False）                    | 原 repo `prismatic/vla/datasets/rlds/traj_transforms.py:44`、`vla-scripts/finetune.py:390`；論文未提及 |

原始實作在 import 時透過檢查命令列參數來決定 chunk 大小與正規化方式
（`prismatic/vla/constants.py`）；本專案則將它們明確定義為 configuration 欄位。

LeRobot 的 `QUANTILES` 與原始實作的 `BOUNDS_Q99` 有兩處不同：原始實作會把正規化
後的值截斷到 `[-1, 1]`（`prismatic/vla/datasets/rlds/utils/data_utils.py:81`），
並且不正規化被 mask 的維度。例如 LIBERO checkpoint 的 `dataset_statistics.json`
把 gripper 的 action 維度設為 mask。這兩處差異都由 processor 處理（§8.2）。

論文 Table IV 中其餘的超參數，由其他元件或訓練指令負責：

| 設定（論文 Table IV） | 值                            | 負責的元件                                                   |
| --------------------- | ----------------------------- | ------------------------------------------------------------ |
| 總 batch size         | 64（每張 GPU 8 × 8 張 GPU）   | `lerobot-train --batch_size`、多 GPU 訓練                    |
| 訓練步數              | 150K（LIBERO-Goal 為 50K）    | `lerobot-train --steps`                                      |
| 輸入影像              | 1 張第三人稱 + 1 張手腕相機   | Dataset features；§2 Vision backbone                         |
| 機器人狀態輸入        | 是                            | Dataset features；§3 Projectors                              |
| 輸入影像尺寸          | 224 × 224                     | §2 Vision backbone（`image_size`）                           |
| LoRA rank             | 32                            | LoRA fine-tuning                                             |
| 影像增強              | 90% random crop、color jitter | 90% crop：§8.3（`image_crop_scale`）；color jitter：訓練指令 |
| FiLM                  | 否                            | 不在範圍內                                                   |

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

### 4. Language model 與 bidirectional attention

`BidirectionalLlama`（`language_model.py`）包裝 `transformers.LlamaModel`，並以
雙向 self-attention 執行。原本自回歸的 OpenVLA 使用 causal mask，每個 action
token 只能看到它前面的 token。Parallel decoding 則需要每個 action 位置都看得到其
他所有位置，因此把 causal mask 換成只遮住 padding token 的 mask。

| 設定            | 值                                                                      | Config 欄位 | 來源                                                                                                                                                                               |
| --------------- | ----------------------------------------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Language model  | Llama-2 7B：32 層、hidden size 4096、32 個 head、MLP 大小 11008、SiLU   | （固定）    | 論文 App. A；原 repo 釋出的 `config.json` 中的 `llm_backbone_id: llama2-7b-pure`                                                                                                   |
| 詞彙表          | 32064（Llama-2 的 32000 個 token + 1 個 pad token，再補齊到 64 的倍數） | （固定）    | 原 repo：釋出的 `config.json` 中的 `text_config.vocab_size`、`pad_to_multiple_of`                                                                                                  |
| Pad token id    | 32000                                                                   | （固定）    | 原 repo：釋出的 `config.json` 中的 `pad_token_id`                                                                                                                                  |
| RMSNorm epsilon | 1e-6                                                                    | （固定）    | 原 repo：`LlamaConfig` 的預設值，因為 `prismatic/extern/hf/configuration_prismatic.py:119-123` 只設定詞彙表與 pad token；論文未提及                                                |
| Attention       | 雙向；只遮住 padding 的 key                                             | （固定）    | 論文 App. A（OFT 修改 3）、App. B.1；原 repo `pyproject.toml:50` 與 transformers fork 的 commit [`bc339d9`](https://github.com/moojink/transformers-openvla-oft/commit/bc339d9ad7) |
| 輸出            | 最後一個 RMSNorm 之後的 hidden state；不使用 language-model head        | （固定）    | 論文 App. A（OFT 修改 4）；原 repo `prismatic/extern/hf/modeling_prismatic.py:913`（`hidden_states[-1]`）                                                                          |
| Attention 後端  | PyTorch SDPA                                                            | （固定）    | 本專案；fork 修改的也是 SDPA 路徑                                                                                                                                                  |

RMSNorm epsilon 為 1e-6，與 Meta 的 Llama-2 7B 設定中的 1e-5 不同（由於 Meta 的
repository 需要申請存取，這裡是透過公開鏡像
[`NousResearch/Llama-2-7b-hf`](https://huggingface.co/NousResearch/Llama-2-7b-hf/blob/main/config.json)
確認）。釋出的 OpenVLA-OFT checkpoint 是以 1e-6 fine-tune 的，因此本專案沿用 1e-6。

與原始實作的差異（本專案）：

- **不使用 transformers fork。** 原始實作依賴一個 `transformers` 4.40.1 的 fork，
  唯一的修改是在 attention 層內改寫 Llama 的 causal mask：把 mask 的最後一列複製
  到每一列。對於右側 padding 的序列，這樣剛好只會遮住 padding 的 key。本專案則明
  確建立這個 mask（`bidirectional_attention_mask`），再傳給原版的 `LlamaModel`。
  `transformers` 5 會直接使用傳入的 4D mask（`transformers/masking_utils.py`：
  「If the mask is already 4D, simply return as-is」），而且不論 padding 在哪一側
  結果都相同。Test 會同時以 SDPA 與 eager 兩種 attention 後端驗證。
- **不使用 language-model head。** 原始實作保留 `LlamaForCausalLM`，會計算
  OpenVLA-OFT 根本用不到的詞彙 logits。本專案改用 `LlamaModel`，省去 131M 個不會
  收到 gradient 的參數。`lm_head` 的權重會在轉換時捨棄。

### 5. 序列排列與 parallel decoding

`OpenVLAOFT`（`model.py`）把前面的元件組成一個網路，在一次 forward pass 中預測整
個 action chunk。它為 language model 建立如下的輸入序列：

```text
[BOS] [影像 patch] [proprio] [prompt] [action placeholder] [EOS] [padding]
  1    256 × 影像數     1      不定        K × D = 56          1
```

Action placeholder 是零向量，每個 chunk 步驟的每個 action 維度各一個，因此彼此只
差在 rotary position。Bidirectional attention（§4）讓每個 placeholder 都能讀到影
像、state、prompt，以及其他 placeholder。`action_hidden_states` 回傳用來解碼 56
個 action 值的最後一層 hidden state，再由 §6 轉換成 action。

| 設定                        | 值                                                         | Config 欄位                           | 來源                                                                                                                                                                               |
| --------------------------- | ---------------------------------------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 影像 patch 與 proprio token | 緊接在 BOS 之後插入，proprio 位於 patch 之後               | （固定）                              | 論文 App. A；原 repo `prismatic/extern/hf/modeling_prismatic.py:458`、`:475`                                                                                                       |
| Action placeholder 數量     | `chunk_size × action_dim`（LIBERO 為 8 × 7 = 56）          | `chunk_size`、action feature 的 shape | 論文 §IV-B；原 repo `prismatic/extern/hf/modeling_prismatic.py:737`                                                                                                                |
| Placeholder embedding       | 零向量                                                     | （固定）                              | 論文 §IV-B、App. B.1（「empty action embeddings that differ only in their positional encoding」）；原 repo `prismatic/extern/hf/modeling_prismatic.py:621`（訓練）、`:891`（推論） |
| Placeholder 之後的 EOS      | Llama 的 `</s>`（id 2）                                    | （固定）                              | 原 repo：訓練時的 prompt 格式 `prismatic/models/backbones/llm/prompting/base_prompter.py:37`、推論 `prismatic/extern/hf/modeling_prismatic.py:742-743`；論文未提及                 |
| 讀取 hidden state 的位置    | 位移一格：從最後一個 prompt token 到倒數第二個 placeholder | （固定）                              | 原 repo：訓練 `vla-scripts/finetune.py:343`、`:377-381`，推論 `prismatic/extern/hf/modeling_prismatic.py:914`；論文未提及                                                          |
| Padding                     | 位於 EOS 之後                                              | （固定）                              | 原 repo `prismatic/util/data_utils.py:113`                                                                                                                                         |

位移一格的讀取方式源自自回歸的 OpenVLA：每個位置的 hidden state 用來預測下一個
token。原始訓練腳本把 hidden state 與 `labels[:, 1:]` 對齊，因此每個 action
placeholder 的值，是由它前一個 token 的 hidden state 解碼而來；最後一個
placeholder 的 hidden state 則從未被讀取。釋出的 checkpoint 就是這樣訓練的，因此
本專案保留相同的讀取方式。

與原始實作的差異（本專案）：

- **不使用假 token 產生 placeholder。** 原始實作在 `input_ids` 中放入真正的
  action token id（訓練）或假的 id 1（推論），embed 之後再乘上由 labels 推導出的
  mask 把它們歸零。本專案直接插入零向量，因此不需要 action token，也不需要
  labels。
- **批次中 prompt 長度可以不同。** Prompt 以右側 padding 的形式傳入；模型把
  placeholder 與 EOS 直接接在每個 prompt 後面，再用 stable sort 把 padding 移到最
  後。因此每個樣本的排列與位置都和單獨執行時相同，test 會驗證這一點。原始的推論
  程式只支援 batch size 1（`prismatic/extern/hf/modeling_prismatic.py:747`）。
- **Dependency injection。** `OpenVLAOFT` 接收已經建立好的 vision backbone 與
  language model，只自行建立 projector。如何依 configuration 建立這些元件由
  policy（§7）決定，test 也因此可以傳入小型版本。

### 6. L1 regression action head

`L1RegressionActionHead`（`action_head.py`）取代 language model 的輸出層。對每
個 chunk 步驟，它把該步驟 `action_dim` 個 token 的 hidden state 串接起來（LIBERO
為 7 × 4096），再用一個 residual MLP 回歸出該步驟正規化後的 action 向量。
`OpenVLAOFT.forward` 現在回傳預測的 chunk，shape 為
`(B, chunk_size, action_dim)`。

```text
LayerNorm → Linear(7·4096 → 4096) → ReLU
→ 2 × [x + ReLU(Linear(LayerNorm(x)))]
→ LayerNorm → Linear(4096 → 7)
```

| 設定       | 值                                                                              | Config 欄位  | 來源                                                                                          |
| ---------- | ------------------------------------------------------------------------------- | ------------ | --------------------------------------------------------------------------------------------- |
| Head 類型  | 4 個 linear 層、ReLU 的 MLP，以 L1 regression 訓練                              | （固定）     | 論文 §IV-B、App. A（OFT 修改 4）、App. B.2；原 repo `prismatic/models/action_heads.py:84-107` |
| 層的結構   | 輸入 LayerNorm 與投影、2 個 pre-LayerNorm residual block、輸出 LayerNorm 與投影 | （固定）     | 原 repo `prismatic/models/action_heads.py:38-81`；論文只寫「4 layers with ReLU activation」   |
| 隱藏層寬度 | 4096，即 language model 的 hidden size                                          | （固定）     | 原 repo `vla-scripts/finetune.py:894`；論文未提及                                             |
| 輸入分組   | 每個步驟的 `action_dim` 個 hidden state 串接                                    | （固定）     | 原 repo `prismatic/models/action_heads.py:95`、`:105`                                         |
| 輸出       | 正規化後的 action，不做壓縮或截斷                                               | （固定）     | 原 repo `prismatic/models/action_heads.py:81`；正規化見 §1                                    |
| Head 大小  | LIBERO 時為 151M 參數                                                           | （推導而得） | 論文 Table IV                                                                                 |

正規化 action 上的 L1 loss（論文 §IV-B、App. D；原 repo
`vla-scripts/finetune.py:390`）由 policy（§7）計算。

與原始實作的差異（本專案）：

- **成為網路的一部分。** 原始實作把 action head 放在 Hugging Face 模型之外，推論
  時再傳入 `predict_action`。本專案把它作為 `OpenVLAOFT` 的子模組，因此單一
  `state_dict` 就包含所有權重。
- **參數名稱。** `MLPResNet` 的 `layer_norm1`、`fc1`、`mlp_resnet_blocks.N.ffn`、
  `layer_norm2`、`fc2`，在本專案中分別命名為 `input_norm`、`input_proj`、
  `blocks.N`、`output_norm`、`output_proj`。Checkpoint 的 key 會在轉換時重新命名。
- **不依賴全域的 chunk 大小。** 原始實作用模組層級的常數 `NUM_ACTIONS_CHUNK` 做
  reshape；本 head 則從輸入推得 chunk 長度。

### 7. Policy

`OpenVLAOFTPolicy`（`modeling_openvla_oft.py`）把 `OpenVLAOFT` 接上 LeRobot 的
`PreTrainedPolicy` 介面，讓 `lerobot-train` 與 `lerobot-eval` 可以透過
`--policy.type openvla_oft` 使用。職責劃分如下：

| 層              | 職責                                                                                      |
| --------------- | ----------------------------------------------------------------------------------------- |
| Processor（§8） | Prompt 模板與 tokenize、影像 resize 與 crop、state 與 action 正規化、action 反正規化      |
| Policy（§7）    | 依 configuration 建立網路、精度設定、把 LeRobot batch 轉成網路輸入、L1 loss、action queue |
| 網路（§2–§6）   | 把 token、影像與 state 映射成正規化的 action chunk                                        |

Policy 讀取以下 batch key，全部由 preprocessor 產生：

| Key                                   | Shape                         | 內容                                                         |
| ------------------------------------- | ----------------------------- | ------------------------------------------------------------ |
| `observation.images.*`                | 每台相機 `(B, 3, H, W)`       | 值域 `[0, 1]` 的影像，依 `config.image_features` 的順序疊起  |
| `observation.state`                   | `(B, state_dim)`              | 正規化後的機器人狀態（可省略）                               |
| `observation.language.tokens`         | `(B, L)`                      | 右側 padding、以 BOS 開頭的 prompt token id                  |
| `observation.language.attention_mask` | `(B, L)`                      | prompt token 為 1，padding 為 0                              |
| `action`                              | `(B, chunk_size, action_dim)` | 正規化後的目標 action（僅訓練時）                            |
| `action_is_pad`                       | `(B, chunk_size)`             | 超出 episode 結尾的步驟（`mask_padded_actions=True` 時使用） |

| 行為            | 值                                                                                 | Config 欄位                       | 來源                                                                                       |
| --------------- | ---------------------------------------------------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------ |
| 訓練 loss       | 正規化 action chunk 上的平均 L1                                                    | `mask_padded_actions`             | 論文 §IV-B、App. D；原 repo `vla-scripts/finetune.py:390`                                  |
| Action 執行方式 | 預測一個 chunk，之後每次呼叫依序回傳前 `n_action_steps` 個 action                  | `n_action_steps`                  | 論文 §V-A、Table IV；原 repo `experiments/robot/libero/run_libero_eval.py:306`、`:328-344` |
| 精度            | 權重使用 `dtype`，proprio projector 可選擇保留 float32，forward 在 autocast 下執行 | `dtype`、`proprio_projector_fp32` | 原 repo，見 §1                                                                             |
| Proprio 輸入    | Dataset 有 `observation.state` 時使用                                              | （由 dataset features 決定）      | 論文 Table IV                                                                              |

補充說明：

- **推論時的精度。** 原始實作訓練時使用 autocast，但評估時以純 bfloat16 執行、不
  使用 autocast，並把 proprio projector 與 action head 轉成 bfloat16
  （`experiments/robot/openvla_utils.py:410`、`:492`）。本專案在訓練與推論都採用
  訓練時的設定。在 autocast 下，float32 的 proprio 權重會在每次矩陣乘法前轉成
  bfloat16，因此計算結果與 bfloat16 權重相同；不過 autocast 會讓 LayerNorm 等部分
  運算以 float32 執行，因此評估時的數值可能與原始實作略有差異。
- **LeRobot 的 `use_amp`。** 請保持 `--policy.use_amp` 關閉。Policy 已經依
  `dtype` 套用 autocast；開啟 `use_amp` 會透過 `accelerate` 再疊加一層混合精度。
- **載入時的記憶體。** 網路先以 float32 建立再轉成 `dtype`，因此建立完整模型時會
  短暫需要約 30 GB 的主機記憶體。這點會在 checkpoint 轉換時一併處理。
- **測試。** `build_model` 是唯一知道完整尺寸架構的地方。Test 會把它換成小型網
  路，因此 configuration 中沒有只為測試而設的欄位。

### 8. Processor

`make_openvla_oft_pre_post_processors`（`processor_openvla_oft.py`）負責建立
policy 前後的兩條 LeRobot pipeline。Preprocessor 把原始觀測（相機影像、機器人狀
態與任務字串）轉成 §7 所描述的 batch；postprocessor 則把 policy 輸出的正規化
action 轉回機器人的 action。以下各小節分別說明其中一個部分。

#### 8.1 Prompt 與 tokenize

任務描述會先轉成小寫，套入 OpenVLA 的 prompt 模板，再用 OpenVLA 的 Llama-2
tokenizer 轉成 token：

```text
In: What action should the robot take to {task}?\nOut: ␣
```

模板結尾有一個空格（以 `␣` 表示）。Llama-2 的 SentencePiece tokenizer 會把這個
結尾空格轉成 token `▁`（id 29871）。原始實作在訓練時把 prompt 與 action 字串一起
tokenize，這個 token 同樣出現在 `Out:` 與第一個 action token 之間；推論時則是組出
不含空格的 prompt，再手動補上 id 29871。在模板中保留這個空格，只要一般的
tokenizer 呼叫就能同時重現這兩種情況。

| 設定                      | 值                                                                     | Config 欄位                              | 來源                                                                                                                                                                                 |
| ------------------------- | ---------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Prompt 模板               | `In: What action should the robot take to {task}?\nOut: `              | （固定）                                 | 原 repo：訓練 `prismatic/vla/datasets/datasets.py:56` 搭配 `prismatic/models/backbones/llm/prompting/base_prompter.py:36`，推論 `experiments/robot/openvla_utils.py:757`；論文未提及 |
| 任務大小寫                | 轉成小寫                                                               | （固定）                                 | 原 repo：訓練 `prismatic/vla/datasets/datasets.py:40`、推論 `experiments/robot/openvla_utils.py:757`                                                                                 |
| 結尾的 `▁` token（29871） | 由結尾空格產生                                                         | （固定）                                 | 原 repo：推論時在 `prismatic/extern/hf/modeling_prismatic.py:972-975` 中手動補上                                                                                                     |
| Tokenizer                 | Llama-2 SentencePiece tokenizer，加上 OpenVLA 的 pad token（id 32000） | `tokenizer_name`（`openvla/openvla-7b`） | 原 repo：釋出 checkpoint 中的 `tokenizer.json`                                                                                                                                       |
| BOS                       | 由 tokenizer 加上（id 1）                                              | （固定）                                 | 原 repo：`prismatic/vla/datasets/datasets.py:63` 中的 `add_special_tokens=True`                                                                                                      |
| Padding                   | 右側，補到 batch 中最長的 prompt                                       | （固定）                                 | 本專案；§5 會把 padding 移到 action placeholder 之後                                                                                                                                 |

驗證方式：用原始環境（`transformers` 4.40.1）與本專案的環境（`transformers`
5.5.4）分別 tokenize 同一個 LIBERO-Spatial 任務的 prompt，兩者的 token id 完全相
同，並由 test `tests/test_processor.py::test_tokens_match_original_tokenizer` 固定下
來。這些 id 也與原始實作在訓練時組出的序列一致（直到第一個 action token 之前）。

實作說明：

- **兩個小 step。** `OpenVLAPromptProcessorStep` 只負責組 prompt，
  `OpenVLATokenizerProcessorStep` 只負責 tokenize。兩者都註冊在 LeRobot 的
  `ProcessorStepRegistry`，因此隨 checkpoint 存下的 pipeline
  （`policy_preprocessor.json`）會記錄它們，之後可以重新載入。
- **為什麼要自訂 tokenizer step。** LeRobot 的 `TokenizerProcessorStep` 用
  `AutoTokenizer` 載入 tokenizer。對 OpenVLA 的 repository，`AutoTokenizer` 會讀
  取 `config.json`，在 `auto_map` 中發現 OpenVLA 的自訂模型程式碼，然後停下來詢問
  是否執行。這個子類別改成直接載入 `LlamaTokenizerFast`，不需要任何自訂程式碼；
  其餘行為（包括 padding 與存檔）都直接繼承。
- **與網路之間的約定。** 網路（§5）要求每個 prompt 以 BOS 開頭、右側 padding，並
  附上 attention mask。Test `tests/test_policy.py::test_accepts_preprocessor_output`
  會讓真正的 preprocessor 輸出直接進入 policy，確保兩邊的約定一致。

#### 8.2 State 與 action 正規化

OpenVLA-OFT 以原始實作稱為 `BOUNDS_Q99` 的方式正規化機器人狀態與 action。對每個
維度，以 dataset 統計值中的第 1 與第 99 百分位數 `q01`、`q99` 計算：

```text
正規化：    x̂ = clip(2 · (x − q01) / (q99 − q01) − 1, −1, 1)
反正規化：  x = (x̂ + 1) / 2 · (q99 − q01) + q01
```

Action 的維度也可以被 mask：被 mask 的維度在正規化與反正規化時都原樣通過。原始實
作會把 end-effector dataset 的 gripper 設為 mask，因為 gripper action 本身已經是
絕對的開合指令，而不是位移量。在釋出的 LIBERO checkpoint 中，gripper action 的值
域是 `0`（閉合）到 `1`（張開），這是原始 data loader 的慣例
（`experiments/robot/robot_utils.py:180-185`），網路也是以這個原始尺度學習輸出。

LeRobot 內建的 `QUANTILES` 只實作了公式的第一部分。
`OpenVLANormalizerProcessorStep` 與 `OpenVLAUnnormalizerProcessorStep` 繼承
LeRobot 的 normalizer 與 unnormalizer，沿用它們管理統計值與存檔的方式，再加上截斷
與 mask。

| 設定                      | 值                                                                           | Config 欄位                            | 來源                                                                                                                                                         |
| ------------------------- | ---------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 值域映射                  | State 與 action 皆為 `[q01, q99]` → `[-1, 1]`                                | `normalization_mapping`（`QUANTILES`） | 原 repo `prismatic/vla/constants.py:30`、`prismatic/vla/datasets/rlds/utils/data_utils.py:72-83`                                                             |
| 截斷                      | 正規化後的值截斷到 `[-1, 1]`：訓練時的 action 目標，以及訓練與推論時的 state | （固定）                               | 原 repo：訓練 `prismatic/vla/datasets/rlds/utils/data_utils.py:81`，推論時的 state `experiments/robot/openvla_utils.py:669-676`                              |
| 反正規化                  | 反向映射，不截斷                                                             | （固定）                               | 原 repo `prismatic/extern/hf/modeling_prismatic.py:785-789`                                                                                                  |
| 被 mask 的 action 維度    | 原樣通過；state 一律不 mask                                                  | `action_norm_mask`                     | 原 repo：end-effector action 的 mask `prismatic/vla/datasets/rlds/oxe/materialize.py:35-39`，套用於 `data_utils.py:79-83` 與 `modeling_prismatic.py:785-789` |
| 預設 mask                 | 無：每個 action 維度都正規化                                                 | `action_norm_mask`                     | 本專案；與原始實作對關節位置 action 的處理一致（`materialize.py:43-45`），適用於 SO-100/SO-101 手臂                                                          |
| LIBERO checkpoint 的 mask | `[True] * 6 + [False]`（gripper 不正規化）                                   | `action_norm_mask`                     | 原 repo：`materialize.py:37-39`；記錄在每個 checkpoint 的 `dataset_statistics.json` 的 `mask` 中                                                             |

為什麼 mask 對 LIBERO checkpoint 很重要：它們的 gripper 統計值是 `q01 = 0`、
`q99 = 1`。若對 gripper 做正規化，會把它映射到 `[-1, 1]`，但網路學到的是在原始的
`[0, 1]` 尺度上輸出，因此反正規化後每個 gripper 指令都會被錯誤解讀。所以轉換
checkpoint 時，會依 `dataset_statistics.json` 設定 `action_norm_mask`。

與原始實作的差異（本專案）：

- **Epsilon。** 原始實作一律在 `q99 − q01` 上加 `1e-8`；LeRobot 只在兩者相等時才
  以 `1e-8` 代替。相對差異約為 `1e-8`，實際上沒有影響。
- **每個 policy 一個 mask。** 原始實作把 mask 存在 dataset 統計值中；本專案則把它
  定義為 configuration 欄位，因此會隨 policy 與 processor 一起存檔，不依賴統計值的
  格式。

驗證方式：`tests/test_processor.py` 把 preprocessor 與 postprocessor 的結果，與直
接照抄原始公式（`data_utils.py:72-83`、`modeling_prismatic.py:785-789`）的計算結果
比對，涵蓋超出 `[q01, q99]` 的值與被 mask 的維度，並確認 mask 在 pipeline 存檔後
重新載入仍然保留。

#### 8.3 影像

影像處理分成 preprocessor 與 policy 兩部分，因為其中一部分在訓練與推論時相同，另
一部分則不同：

```text
相機影像 ─▶ [preprocessor] resize 到 224 × 224
         ─▶ [policy] crop 90% 的面積，再 resize 回 224 × 224
                     訓練：位置隨機的正方形框
                     推論：置中的正方形框
         ─▶ vision backbone（§2）
```

原始實作在 fine-tune 時使用涵蓋 90% 影像面積的隨機 crop，評估時則使用對應的中央
crop，讓網路看到的視野始終一致（論文 Table IV；原始 `LIBERO.md:82` 也因此強調評估時必須
設定 `--center_crop True`）。LeRobot 在兩種模式下執行同一條 preprocessor，因此 crop 放
在 policy 中，依 `self.training` 切換。LeRobot 自己的 Diffusion Policy 也是這樣處
理隨機與中央 crop。

| 設定          | 值                                                   | Config 欄位               | 來源                                                                                                                                                                |
| ------------- | ---------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Resize        | 到 `image_size` × `image_size`，antialias 的 bicubic | `image_size`              | 原 repo：訓練（`dlimp/utils.py` 的 `resize_image`）與評估（`experiments/robot/openvla_utils.py:538-541`）皆使用 Lanczos3；bicubic 是本專案在 PyTorch 中最接近的選項 |
| Crop 面積     | 影像面積的 90%，正方形                               | `image_crop_scale`（0.9） | 論文 Table IV（`random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0])`）；原 repo `prismatic/vla/datasets/datasets.py:152-153`                               |
| 訓練時的 crop | 隨機位置，以 bilinear 取樣 resize 回原尺寸           | `image_crop_scale`        | 原 repo：`dlimp/augmentations.py`（`random_resized_crop`），每台相機使用不同的 seed（`prismatic/vla/datasets/rlds/obs_transforms.py:27-38`）                        |
| 推論時的 crop | 置中，以 bilinear 取樣 resize 回原尺寸               | `image_crop_scale`        | 原 repo `experiments/robot/openvla_utils.py:546-593`（`crop_and_resize`），套用於每台相機（`:682-708`）                                                             |
| 相機          | 每台相機的影像，包括手腕相機                         | （固定）                  | 原 repo：同上                                                                                                                                                       |

要與原始實作完全一致，需要注意的細節：

- **Crop 的取樣方式。** 原始實作的兩種 crop 都使用 `tf.image.crop_and_resize`，框
  的角點可以是小數。`image_crop.py` 中的 `crop_and_resize` 用
  `torch.nn.functional.grid_sample`（`align_corners=True`）重現它的 bilinear 取
  樣；`tests/test_image_crop.py` 把結果與直接照抄 TensorFlow 公式的計算比對。
- **沿對角線移動的隨機 crop。** `random_resized_crop` 用同一個隨機 seed 抽出垂直與
  水平的位移，因此對正方形的框來說兩個位移相等，框只會沿著影像的對角線移動。
  `crop_boxes` 也重現了這一點。

與原始實作的已知差異（本專案）：

- **插值方式。** 原始實作以 TensorFlow 的 antialias Lanczos3 濾波器 resize，評估時
  還會先把每張影像做一次 JPEG 編碼再解碼，以模仿壓縮過的訓練資料。PyTorch 沒有適
  用於 tensor 的 Lanczos resize，因此本專案使用 antialias 的 bicubic，並省略 JPEG
  編碼與解碼。相機影像若已經是目標尺寸，就不做 resize。
- **不做 8 位元捨入。** 原始實作在正規化前會把 crop 後的影像轉回 8 位元整數；本專
  案則保持浮點數。
- **Color jitter。** 原始實作在訓練時也會擾動亮度、對比、飽和度與色相（論文 Table
  IV）。LeRobot 的 dataset 已經提供這些轉換（`--dataset.image_transforms`），因此
  在訓練指令中設定，而不是在這裡。

這些差異只會讓像素值略有不同。若 LIBERO 的成功率低於論文，這裡是第一個要檢查的地
方。
