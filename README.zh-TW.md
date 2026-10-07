# lerobot-policy-openvla-oft

[English](README.md) | 繁體中文

本專案是 [LeRobot](https://github.com/huggingface/lerobot) 的 out-of-tree
policy 插件，將 [OpenVLA-OFT](https://openvla-oft.github.io/)（Kim、Finn 與
Liang，2025）移植到 LeRobot v0.6.1，以 `--policy.type openvla_oft` 使用。插件
可將原作者釋出的官方 LIBERO checkpoint 轉換為 LeRobot 格式，並以
`lerobot-eval` 評估，目標是復現論文中的 LIBERO 結果。此外也支援以
`lerobot-train` 從 `openvla/openvla-7b` 進行 LoRA fine-tune，並以
`lerobot-rollout` 將 fine-tune 後的 policy 部署到單臂 SO-100 或 SO-101
follower 手臂上。OpenVLA-OFT+ 的 FiLM 語言條件化則作為選用功能提供（§12）。

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
| 編譯                           | 關閉                                | `compile_model`、`compile_mode`（`default`）      | 本專案；比照 LeRobot 的 pi0 與 SmolVLA（`compile_model`、`compile_mode`）                              |
| FiLM（OpenVLA-OFT+）           | 關閉                                | `use_film`                                        | 論文 App. D、Table IV；原 repo `vla-scripts/finetune.py:83`；見 §12                                    |
| FiLM 平均是否包含 padding      | 包含                                | `film_mask_padding`（False）                      | 原 repo `prismatic/extern/hf/modeling_prismatic.py:581`；論文未提及；見 §12                            |

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
| FiLM                  | 否                            | §12 FiLM（`use_film`）                                       |

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
- **Buffer 維持 float32。** 只有參數會轉成 `dtype`；buffer 維持建立時的 dtype：
  Llama 的 RoPE 頻率（`inv_freq`）與 vision backbone 的像素統計值都維持 float32。
  原始實作也把 RoPE 頻率維持在 float32：`LlamaRotaryEmbedding` 以 `.float()` 計算
  它們（transformers fork 的 `modeling_llama.py:103`），模型以
  `torch_dtype=torch.bfloat16` 載入後也不會再轉換整個模型。若把頻率轉成 bfloat16，
  在序列尾端附近的位置 600，旋轉角度最多會偏移 0.7 rad；以 libero-spatial
  checkpoint 實測，正規化後的 action 平均改變 0.0018（約為其大小的 1%），最大
  0.012。
- **Apple MPS 上的 RoPE。** `transformers` 只在 CPU 與 CUDA 上於計算 RoPE 時關閉
  autocast。在 MPS 上，policy 的 autocast 仍然有效，因此旋轉角度（包括位置）會以
  bfloat16 計算。這只影響在 Mac 上執行 policy；CUDA 上的評估與訓練都以 float32 計
  算 RoPE。
- **編譯。** 設定 `--policy.compile_model=true` 時，policy 會以 `torch.compile` 編譯網路
  （模式由 `--policy.compile_mode` 選擇）。它使用 `nn.Module.compile`，就地編譯模組
  的呼叫，而不是把模組包起來，因此參數名稱、checkpoint 與 LoRA 的套用對象都不會改變；第
  一次呼叫時才會 trace 網路，此時 `lerobot-train` 已經加上 LoRA adapter。網路會編譯
  成單一 graph，沒有 graph break。由於 prompt 會補齊到 batch 中最長的那一個，出現
  第二種 prompt 長度時會以 dynamic shape 重新編譯一次，之後就不會再重新編譯。第一
  次呼叫需要負擔編譯時間，因此短時間的執行整體可能反而變慢；Inductor 對 MPS 的支
  援也有限。CUDA 上的加速幅度尚未量測。
- **LeRobot 的 `use_amp`。** 請保持 `--policy.use_amp` 關閉。Policy 已經依
  `dtype` 套用 autocast；開啟 `use_amp` 會透過 `accelerate` 再疊加一層混合精度。
- **載入時不做隨機初始化。** `OpenVLAOFTPolicy.from_pretrained` 建立網路時，把參數
  放在 PyTorch 的 meta device 上，因此不占記憶體，也不做隨機初始化；接著直接採用
  checkpoint 的 tensor 作為參數（`load_state_dict(..., assign=True)`），放在目標裝
  置上並轉成各參數的 dtype。Buffer 照常建立，因為 checkpoint 不含 buffer。
  Checkpoint 中沒有的參數，例如 base OpenVLA 模型的 action head 與 proprio
  projector，之後再以 PyTorch 的預設方式初始化。以 `--policy.type` 從頭訓練時，仍
  照常建立並初始化網路。在 Apple M5 Max 上，載入 libero-spatial checkpoint 從 55.2
  秒、主機記憶體峰值 54 GB，降到 3.1 秒、30.6 GB，參數與 buffer 完全相同；剩下的峰
  值大多是以 memory map 讀入的 checkpoint 檔案。多 GPU 訓練時每個 process 都會各自
  載入一份，因此這點在多 GPU 時最為重要。
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

Pipeline 以 LeRobot 本身的 step 再加上一個小 step 組成：

```text
preprocessor：  … → normalizer_processor（QUANTILES） → openvla_oft_clip
postprocessor： unnormalizer_processor（QUANTILES） → …
```

LeRobot 的 `QUANTILES` 就是公式的第一部分，`OpenVLAClipProcessorStep` 再加上截
斷。保留 LeRobot 的 normalizer 與 unnormalizer 而不另外取代，對 fine-tune 很重
要：`lerobot-train` 會依名稱找出 `normalizer_processor` 與
`unnormalizer_processor`，把 dataset 統計值注入預訓練 policy 的 processor
（`lerobot/scripts/lerobot_train.py:357-377`）。名稱不同的 step 會在不報錯的情況
下沿用預訓練 policy 的統計值。

**不正規化的維度。** 原始實作也可以讓某些 action 維度不做正規化：它會把
end-effector dataset 的 gripper 設為 mask，因為 gripper action 本身已經是絕對的開
合指令，而不是位移量。在釋出的 LIBERO checkpoint 中，gripper action 的值域是
`0`（閉合）到 `1`（張開），這是原始 data loader 的慣例
（`experiments/robot/robot_utils.py:180-185`），網路也是以這個原始尺度學習輸出。本
專案不另外實作 mask 機制，而是透過統計值表達：`q01 = -1`、`q99 = 1` 時，
`QUANTILES` 會把 `x` 映射成 `2 · (x + 1) / 2 − 1 = x`，也就是恆等映射。Checkpoint
轉換（§9）會對 `dataset_statistics.json` 中 `mask` 為 False 的每個維度寫入這組百分
位數。

| 設定                             | 值                                                                                          | Config 欄位                            | 來源                                                                                                                                                         |
| -------------------------------- | ------------------------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 值域映射                         | State 與 action 皆為 `[q01, q99]` → `[-1, 1]`                                               | `normalization_mapping`（`QUANTILES`） | 原 repo `prismatic/vla/constants.py:30`、`prismatic/vla/datasets/rlds/utils/data_utils.py:72-83`                                                             |
| 截斷                             | 正規化後的 state 與 action 截斷到 `[-1, 1]`：訓練時的 action 目標，以及訓練與推論時的 state | （固定）                               | 原 repo：訓練 `prismatic/vla/datasets/rlds/utils/data_utils.py:81`，推論時的 state `experiments/robot/openvla_utils.py:669-676`                              |
| 反正規化                         | 反向映射，不截斷                                                                            | （固定）                               | 原 repo `prismatic/extern/hf/modeling_prismatic.py:785-789`                                                                                                  |
| 被 mask 的 action 維度           | 恆等映射，透過 `q01 = -1`、`q99 = 1`                                                        | （統計值）                             | 原 repo：end-effector action 的 mask `prismatic/vla/datasets/rlds/oxe/materialize.py:35-39`，套用於 `data_utils.py:79-83` 與 `modeling_prismatic.py:785-789` |
| 新訓練時被 mask 的維度           | 無：每個維度都使用 dataset 統計值                                                           | （統計值）                             | 本專案；與原始實作對關節位置 action 的處理一致（`materialize.py:43-45`），適用於 SO-100/SO-101 手臂                                                          |
| LIBERO checkpoint 被 mask 的維度 | Gripper                                                                                     | （統計值）                             | 原 repo：`materialize.py:37-39`；記錄在每個 checkpoint 的 `dataset_statistics.json` 的 `mask` 中                                                             |

為什麼 mask 對 LIBERO checkpoint 很重要：它們的 gripper 統計值是 `q01 = 0`、
`q99 = 1`。若以這組統計值對 gripper 做正規化，會把它映射到 `[-1, 1]`，但網路學到的
是在原始的 `[0, 1]` 尺度上輸出，因此反正規化後每個 gripper 指令都會被錯誤解讀。

與原始實作的差異（本專案）：

- **被 mask 的維度也會截斷。** 原始實作讓被 mask 的維度原樣通過、不截斷；本專案
  在恆等映射之後，與其他維度一樣截斷。這只會影響 `[-1, 1]` 以外的值，而 LIBERO
  的 gripper action 只有 0 或 1。
- **Epsilon。** 原始實作一律在 `q99 − q01` 上加 `1e-8`；LeRobot 只在兩者相等時才
  以 `1e-8` 代替。相對差異約為 `1e-8`，實際上沒有影響。

驗證方式：

- `tests/test_processor.py` 把 preprocessor 與 postprocessor 的結果，與直接照抄原
  始公式（`data_utils.py:72-83`、`modeling_prismatic.py:785-789`）的計算比對，涵
  蓋超出 `[q01, q99]` 的值，並確認存檔的 pipeline 使用 LeRobot 的
  `normalizer_processor` 與 `unnormalizer_processor`。
- `tests/test_convert_checkpoint.py` 確認被 mask 的 gripper 值在兩個方向都原樣通
  過。
- 與本節先前的實作（自訂 normalizer 加上明確的 mask）相比，以 libero-spatial 的統
  計值、64 組隨機輸入（包含超出範圍的值）測試，正規化後的 state 與反正規化後的
  action 完全相同；正規化後的 action 目標只在 gripper 原始值大於 1 時不同，也就是
  上述的截斷差異。

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

### 9. Checkpoint 轉換

`convert_checkpoint.py` 把釋出的 OpenVLA-OFT checkpoint 轉成 LeRobot 的 policy 目
錄，可以直接用 `--policy.path` 載入：

```bash
uv run python -m lerobot_policy_openvla_oft.convert_checkpoint \
    --repo-id moojink/openvla-7b-oft-finetuned-libero-spatial \
    --output-dir outputs/checkpoints/libero-spatial
```

`outputs/` 已加入 `.gitignore`。腳本只會下載需要的檔案（每個 checkpoint 約
15.5 GB）到 Hugging Face cache，並輸出約 15 GB 的 bfloat16 權重與 processor
pipeline。

釋出的 checkpoint：

| Repository                                                       | 訓練資料                 |
| ---------------------------------------------------------------- | ------------------------ |
| `moojink/openvla-7b-oft-finetuned-libero-spatial`                | LIBERO-Spatial           |
| `moojink/openvla-7b-oft-finetuned-libero-object`                 | LIBERO-Object            |
| `moojink/openvla-7b-oft-finetuned-libero-goal`                   | LIBERO-Goal              |
| `moojink/openvla-7b-oft-finetuned-libero-10`                     | LIBERO-10（LIBERO-Long） |
| `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` | 四個 suite 合併          |

每個釋出檔案的轉換方式：

| 釋出檔案                                | 內容                                                            | 轉換方式                                                                                                         |
| --------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `model-0000{1..4}-of-00004.safetensors` | Vision backbone、vision projector 與 Llama-2，LoRA 權重已經合併 | 依下表重新命名 key；捨棄不會用到的權重                                                                           |
| `action_head--*.pt`                     | L1 regression action head（§6）                                 | 依下表重新命名 key                                                                                               |
| `proprio_projector--*.pt`               | Proprio projector（§3）                                         | 移除 `module.` 前綴                                                                                              |
| `dataset_statistics.json`               | `q01`、`q99` 與 action 的 `mask`                                | 轉成 `observation.state` 與 `action` 的 LeRobot 統計值；被 mask 的 action 維度設為 `q01 = -1`、`q99 = 1`（§8.2） |
| `lora_adapter/`                         | 未合併的 LoRA 權重                                              | 不需要，因為模型權重已經包含它們                                                                                 |

Key 對應：

| 釋出的 key                                                                | 本專案                                               | 原因                             |
| ------------------------------------------------------------------------- | ---------------------------------------------------- | -------------------------------- |
| `vision_backbone.featurizer.*`                                            | `model.vision.dinov2.vit.*`                          | §2                               |
| `vision_backbone.fused_featurizer.*`                                      | `model.vision.siglip.vit.*`                          | §2                               |
| `*.ls1.scale_factor`、`*.ls2.scale_factor`                                | `*.ls1.gamma`、`*.ls2.gamma`                         | timm 的 LayerScale 名稱（§2）    |
| `projector.*`                                                             | `model.vision_projector.*`                           | §3                               |
| `language_model.model.*`                                                  | `model.llm.model.*`                                  | §4                               |
| `module.*`（proprio projector）                                           | `model.proprio_projector.*`                          | §3                               |
| `module.model.layer_norm1`、`fc1`                                         | `model.action_head.input_norm`、`input_proj`         | §6                               |
| `module.model.mlp_resnet_blocks.N.ffn.0`、`ffn.1`                         | `model.action_head.blocks.N.norm`、`blocks.N.linear` | §6                               |
| `module.model.layer_norm2`、`fc2`                                         | `model.action_head.output_norm`、`output_proj`       | §6                               |
| `language_model.lm_head.weight`                                           | 捨棄                                                 | 不使用 language-model head（§4） |
| `vision_backbone.featurizer.blocks.23.*`、`.norm.*`                       | 捨棄                                                 | DINOv2 中不會用到的層（§2）      |
| `vision_backbone.fused_featurizer.blocks.26.*`、`.norm.*`、`.attn_pool.*` | 捨棄                                                 | SigLIP 中不會用到的層（§2）      |

隨每個 checkpoint 寫出的 configuration 對齊 LeRobot 的 LIBERO 環境：第三人稱相機
`observation.images.image` 在前、手腕相機 `observation.images.image2` 在後，與原
始實作的順序相同（`prismatic/vla/datasets/rlds/oxe/configs.py:645-651`），接著是
8 維的 `observation.state` 與 7 維的 `action`。

驗證方式：

- **嚴格載入。** `load_released_weights` 會對應每一個釋出的 key，並以
  `strict=True` 載入。只有上表列為捨棄的權重可以剩下；其他任何對應不到或缺少的
  key 都會報錯。
- **離線 key 檢查。** 以釋出的 `model.safetensors.index.json` 檢查，剩下的 981 個
  key 涵蓋本專案全部 938 個 vision、projector 與 language model 參數，另外 43 個正
  好是被移除的 vision 層。
- **Test。** `tests/test_convert_checkpoint.py` 從小型模型產生釋出格式的
  state dict，再轉換回來，要求每個值完全相同；另外也測試未知 key 與缺少 key 的失
  敗情況。
- **實際轉換。** 在 Apple M5 Max 上以 CPU 轉換 `libero-spatial` 需 10.6 秒，主機記
  憶體峰值為 15.6 GB（改為不做隨機初始化之前為 100 秒、38 GB，見 §7；輸出逐位元相
  同）。抽查的 8 個 tensor（DINOv2、SigLIP、vision projector、
  Llama-2、action head、proprio projector）與釋出檔案逐位元相同，包括 float32 的
  proprio projector。以 `from_pretrained` 重新載入後，policy 在 MPS 上以 5.7 秒預
  測出數值有限、shape 為 `(1, 8, 7)` 的 action chunk。

釋出的 `.pt` 檔是從 CUDA tensor 存下來的，因此腳本以 `map_location="cpu"` 載入。

### 10. LIBERO 評估

轉換後的 checkpoint 以 `lerobot-eval` 在 LeRobot 內建的 LIBERO 環境中評估。下表逐
項比較該環境與原始評估腳本（`experiments/robot/libero/run_libero_eval.py` 與
`libero_utils.py`）。

| 項目               | 原始實作                                                                             | LeRobot v0.6.1                                                                                                                    | 負責處理                                |
| ------------------ | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| 影像旋轉           | 180°（`libero_utils.py:36`、`:43`）                                                  | `LiberoProcessorStep` 旋轉 180°                                                                                                   | LeRobot                                 |
| 相機               | 第三人稱在前、手腕在後                                                               | `image` 在前、`image2` 在後                                                                                                       | 轉換時的 config（§9）                   |
| State（8 維）      | End-effector 位置、axis-angle 姿態、gripper 關節位置（`run_libero_eval.py:257-259`） | `LiberoProcessorStep` 以相同方式組成                                                                                              | LeRobot                                 |
| 等待物體穩定       | 10 步 no-op `[0, 0, 0, 0, 0, 0, -1]`（`run_libero_eval.py:318-321`）                 | 相同，在 `reset()` 中執行                                                                                                         | LeRobot                                 |
| 初始狀態           | 每個任務 50 個固定狀態，每次試驗用一個（`run_libero_eval.py:227-230`）               | 每個子環境依序使用相同的固定狀態                                                                                                  | LeRobot                                 |
| 成功判定           | `env.step` 回傳的 `done`                                                             | `done` 或 `check_success()`                                                                                                       | LeRobot                                 |
| Action chunking    | 執行完 8 個 action 再重新預測（`run_libero_eval.py:306-344`）                        | Policy 的 action queue                                                                                                            | Policy（§7）                            |
| **Gripper action** | `[0, 1]` → `[-1, 1]`，二值化後取負（`run_libero_eval.py:265-274`）                   | 原樣傳給環境                                                                                                                      | **`OpenVLALiberoGripperProcessorStep`** |
| **渲染解析度**     | 256 × 256（`run_libero_eval.py:116`）                                                | 預設 360 × 360                                                                                                                    | **評估參數**                            |
| **Episode 長度**   | Spatial 220、Object 280、Goal 300、Long 520（`run_libero_eval.py:63-69`）            | Spatial 為 280，其餘相同                                                                                                          | **Spatial 的評估參數**                  |
| **MuJoCo**         | （原始環境）                                                                         | `mujoco >= 3.4` 會破壞 LIBERO-Spatial 第 5 個任務的初始狀態（[lerobot#4390](https://github.com/huggingface/lerobot/issues/4390)） | **評估環境**                            |
| 環境 seed          | 固定 `env.seed(0)`（`libero_utils.py:24`）                                           | 每次 reset 以 `--seed` 設定                                                                                                       | 未對齊；影響很小                        |

**Gripper 轉換。** 原始 data loader 把 gripper action 存成 0（閉合）到 1（張開），
釋出的 checkpoint 也以這個尺度預測（§8.2）。LIBERO 則需要 -1（張開）或 +1（閉
合）。因此原始評估會把預測值映射到 `[-1, 1]`、取正負號，再取負
（`experiments/robot/robot_utils.py:149-198`）。LeRobot 只讓 X-VLA 注入 LIBERO 專
屬的 action 處理（`lerobot/envs/factory.py`），所以本專案改把這個轉換放進轉換後的
checkpoint：`convert_checkpoint.py` 會把 `OpenVLALiberoGripperProcessorStep` 加到
它們的 postprocessor 最後。這個 step 屬於這批 LIBERO checkpoint，而不屬於
policy；若 policy 以已經採用 LIBERO 慣例的資料訓練，就不需要它。

**評估指令。** 評估一個 suite，並對齊原始的評估方式：

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

只有 LIBERO-Spatial 需要 `--env.episode_length`；其他 suite 本來就使用原始的上限。
`--eval.n_episodes=50` 讓一個 suite 的 10 個任務各執行 50 次，與論文一樣每個
suite 共 500 次試驗。

**評估環境。** LeRobot 的 `libero` extra 只能安裝在 Linux 上（`hf-libero` 標記為
`sys_platform == 'linux'`），因此評估需要 Linux 機器。由於 lerobot#4390，該環境
需固定 `mujoco<3.4`。

**目標數字。** 論文中的成功率，每個 suite 為 500 次試驗的平均，依原始
`LIBERO.md` 的說明另外對三個 seed 取平均：

| Checkpoint                 | Spatial | Object | Goal | Long | 平均 | 來源           |
| -------------------------- | ------- | ------ | ---- | ---- | ---- | -------------- |
| 每個 suite 各一個 policy   | 97.6    | 98.4   | 97.9 | 94.5 | 97.1 | 論文 Table I   |
| 四個 suite 共用一個 policy | 97.7    | 98.0   | 96.1 | 95.3 | 96.8 | 論文 Table XIV |

本專案需要接近到什麼程度，等有結果後再決定。已知的差異來源列在 §7（推論精度）與
§8.3（影像插值）。

驗證方式：`tests/test_processor.py` 把 `OpenVLALiberoGripperProcessorStep` 與照抄
原始 `normalize_gripper_action`、`invert_gripper_action` 的計算比對；
`tests/test_convert_checkpoint.py` 確認 LIBERO 的 postprocessor 會在反正規化之後套用
它。

#### 結果

這些結果是在修正 RoPE 頻率維持 float32 之前取得的（§7「Buffer 維持 float32」），
當時頻率被轉成了 bfloat16。之後尚未重新執行。

四個 suite 都以釋出的單一 suite checkpoint 評估，各跑一個 seed（`--seed=7`，每個
suite 500 個 episode）。論文則是三個 seed 的平均。

| Suite          | Checkpoint       | 本專案（%） | 論文 Table I（%） | 差距      |
| -------------- | ---------------- | ----------- | ----------------- | --------- |
| LIBERO-Spatial | `libero-spatial` | 98.0        | 97.6              | +0.4      |
| LIBERO-Object  | `libero-object`  | 98.0        | 98.4              | -0.4      |
| LIBERO-Goal    | `libero-goal`    | 97.2        | 97.9              | -0.7      |
| LIBERO-Long    | `libero-10`      | 94.2        | 94.5              | -0.3      |
| **平均**       |                  | **96.85**   | **97.1**          | **-0.25** |

以 500 個 episode 計，成功率在 97% 附近時，單一數字的 95% 信賴區間約為 ±1.5 個
百分點，因此每個 suite 都在論文數字的抽樣誤差範圍內。

**LIBERO-Spatial** (`moojink/openvla-7b-oft-finetuned-libero-spatial`)

| 任務 | 指令                                                                                     | 成功                |
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
|      | **合計**                                                                                 | **490/500 (98.0%)** |

**LIBERO-Object** (`moojink/openvla-7b-oft-finetuned-libero-object`)

| 任務 | 指令                                                     | 成功                |
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
|      | **合計**                                                 | **490/500 (98.0%)** |

**LIBERO-Goal** (`moojink/openvla-7b-oft-finetuned-libero-goal`)

| 任務 | 指令                                        | 成功                |
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
|      | **合計**                                    | **486/500 (97.2%)** |

**LIBERO-Long** (`moojink/openvla-7b-oft-finetuned-libero-10`)

| 任務 | 指令                                                                                    | 成功                |
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
|      | **合計**                                                                                | **471/500 (94.2%)** |

執行細節（2026-10-05）：

| 項目     | 值                                                                                                                                                                                               |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 指令     | 先以 `sky launch -c openvla-oft-eval skypilot/libero_eval.yaml -i 15 --down` 跑 2 個 episode 的 smoke test，再對每個 suite 各執行一次 `sky exec`，設定 `CHECKPOINT`、`SUITE` 與 `EPISODE_LENGTH` |
| 程式碼   | `db72470`（Spatial）、`857fd66`（Object、Goal、Long；只修改了 SkyPilot 設定）                                                                                                                    |
| 機器     | RunPod L40S（48 GB）、12 vCPU、62 GB RAM、EU-NL-1 機房，目錄價每小時 $1.09；當時所有機房都沒有 L40                                                                                               |
| 軟體     | Python 3.12、PyTorch 2.11.0+cu130、transformers 5.5.4、LeRobot 0.6.1、MuJoCo 3.3.7、robosuite 1.4.0、`MUJOCO_GL=egl`                                                                             |
| 評估設定 | 同時 10 個環境（`--eval.batch_size=10`）、256 × 256 渲染、原始的 episode 長度上限                                                                                                                |

| Suite          | Episode 上限 | 評估時間  | 每個 episode（同時 10 個） |
| -------------- | ------------ | --------- | -------------------------- |
| LIBERO-Spatial | 220          | 26.1 分鐘 | 3.1 秒                     |
| LIBERO-Object  | 280          | 30.8 分鐘 | 3.7 秒                     |
| LIBERO-Goal    | 300          | 27.5 分鐘 | 3.3 秒                     |
| LIBERO-Long    | 520          | 57.5 分鐘 | 6.9 秒                     |

| 階段                                                    | 時間                                                                                                                                             |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| 開啟 L40S                                               | 約 2 分鐘                                                                                                                                        |
| 新機器上的 setup（`apt-get`、`uv sync --extra libero`） | 約 1 分鐘                                                                                                                                        |
| 從 Hugging Face Hub 下載 checkpoint（15.5 GB）          | 每個 checkpoint 約 1 到 2 分鐘                                                                                                                   |
| 轉換 checkpoint                                         | 每個 checkpoint 約 2 分鐘                                                                                                                        |
| 每個 suite 的整個 job（含下載、轉換與載入模型）         | Spatial 28.3 分鐘、Object 35.4 分鐘、Goal 32.0 分鐘、Long 62.0 分鐘                                                                              |
| 機器總使用時間                                          | 約 3 小時 4 分鐘（15:30 至 18:34 UTC），包含兩次修正 setup 與 2 個 episode 的 smoke test；以目錄價計約 $3.3。最後一個 suite 結束後立即關閉機器。 |

成功率最低的任務是 LIBERO-Long 任務 8（78%，把兩個摩卡壺放上爐子）、LIBERO-Goal
任務 3（86%）與 LIBERO-Spatial 任務 5（90%）。Spatial 任務 5 正是 lerobot#4390 影
響的任務；本次使用 MuJoCo 3.3.7，低於受影響的 3.4 以上版本。

### 11. LoRA fine-tuning

Fine-tune 採用 LeRobot 本身的 PEFT 流程
（[PEFT training guide](https://huggingface.co/docs/lerobot/peft_training)）：
`lerobot-train` 從一個預訓練的 LeRobot policy（`--policy.path`）開始，套上 LoRA
adapter（`--peft.*`），只儲存 adapter 與完整訓練的模組。對 OpenVLA-OFT 而言，這個
預訓練 policy 就是以 `convert_checkpoint.py --base` 轉換的 base OpenVLA 模型。

```bash
# 1. 轉換一次 base OpenVLA 模型（約 15 GB）。
uv run python -m lerobot_policy_openvla_oft.convert_checkpoint --base \\
    --repo-id openvla/openvla-7b \\
    --output-dir outputs/checkpoints/openvla-7b

# 2. 以 8 張 GPU 做 LoRA fine-tune（每張 8 筆，共 64 筆）。
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

| 設定                 | 值                                                                                                  | 設定方式                                     | 來源                                                                            |
| -------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------- |
| 起點                 | `openvla/openvla-7b`                                                                                | `--policy.path` 指向轉換後的 base            | 原 repo `vla-scripts/finetune.py:71`（`vla_path` 的預設值），於 `:835-837` 載入 |
| LoRA rank            | 32                                                                                                  | `--peft.r=32`                                | 論文 Table IV；原 repo `vla-scripts/finetune.py:107`                            |
| LoRA alpha           | 16                                                                                                  | Policy 預設值                                | 原 repo `vla-scripts/finetune.py:849`（`min(rank, 16)`）；論文未提及            |
| LoRA dropout         | 0                                                                                                   | Policy 預設值                                | 原 repo `vla-scripts/finetune.py:108`；論文未提及                               |
| LoRA 初始化          | Gaussian                                                                                            | Policy 預設值                                | 原 repo `vla-scripts/finetune.py:852`；論文未提及                               |
| LoRA 套用對象        | Vision backbone、vision projector 與 language model 的所有 linear 層                                | Policy 預設值（`LORA_TARGET_MODULES`）       | 原 repo `vla-scripts/finetune.py:851`（`all-linear`）                           |
| 完整訓練的模組       | Action head（151M）與 proprio projector（17M）；開啟 `use_film` 時另含 FiLM projection（438M，§12） | Policy 預設值（`modules_to_save`）           | 論文 Table IV；原 repo `vla-scripts/finetune.py:876-896`、`:927-933`            |
| Batch size           | 每張 GPU 8 筆，8 張 GPU 共 64 筆                                                                    | `--batch_size=8` 搭配 8 個 process           | 論文 Table IV；原 repo `vla-scripts/finetune.py:88`                             |
| Learning rate 與衰減 | 5e-4，100K 步後 ×0.1                                                                                | Policy 的預設設定（§1）                      | 論文 Table IV、App. D                                                           |
| 訓練步數             | 150K（LIBERO-Goal 為 50K）                                                                          | `--steps`                                    | 論文 Table IV                                                                   |
| 隨機 crop            | 影像面積的 90%                                                                                      | Policy（§8.3）                               | 論文 Table IV                                                                   |
| Color jitter         | Brightness、contrast `[0.8, 1.2]`、saturation `[0.8, 1.2]`、hue `[-0.05, 0.05]`，依此順序全部套用   | `--dataset.image_transforms.*`               | 論文 Table IV；原 repo `prismatic/vla/datasets/datasets.py:152-165`             |
| 混合精度             | 由 policy 進行 bfloat16 autocast                                                                    | 不要對 `accelerate` 傳入 `--mixed_precision` | §7                                                                              |

LoRA 參數量為什麼與論文相符：對每個目標層使用 rank 32，adapter 共有 107.9M 個參
數。原始實作對整個模型套用 `all-linear`，包括本專案移除的 vision 層（§2）與
language-model head（§4）；把這些算進去是 110.8M，也就是論文 Table IV 中的「111M
LoRA adapter」。`tests/test_peft.py` 會確認 `LORA_TARGET_MODULES` 選中的正好是完整尺
寸網路中的所有 linear 層。

與 LeRobot 的銜接方式：

- **不含 features 的 base checkpoint。** 轉換後的 base 中 `input_features` 與
  `output_features` 都是空的，因此 `lerobot-train` 會從 dataset 取得。Base 不含
  action head 與 proprio projector；LeRobot 以非嚴格模式載入 checkpoint，所以這兩
  個模組會保留 PyTorch 的初始化，與原始實作相同（`vla-scripts/finetune.py:876-896`）。
- **Dataset 統計值。** 隨 base 存下的 processor 使用 LeRobot 的
  `normalizer_processor` 與 `unnormalizer_processor`（§8.2），`lerobot-train` 會把
  dataset 統計值注入其中。
- **PEFT 預設值。** `OpenVLAOFTPolicy._get_default_peft_targets()` 提供套用對象、完整
  訓練的模組、alpha、dropout 與初始化方式。LeRobot 的 CLI 一定會傳入 `--peft.r`
  （預設 16），因此 rank 必須在指令中設定。
- **Checkpoint。** 每個 checkpoint 包含 LoRA adapter，以及 action head 與 proprio
  projector；它的 adapter 設定會記錄 base 目錄的路徑。評估 fine-tune 後的 policy
  時，把它的 checkpoint 傳給 `--policy.path`；LeRobot 會從記錄的路徑載入 base，因此
  base 目錄必須存在於相同的位置。

與原始實作的差異（本專案）：

- **Brightness jitter。** 原始實作是加上 `[-0.2, 0.2]` 之間的隨機位移
  （`dlimp/augmentations.py` 中的 `tf.image.stateless_random_brightness`）；
  torchvision 的 `ColorJitter` 則是乘上 `[0.8, 1.2]` 之間的倍數。
- **增強的順序。** 原始實作先 crop 再做 color jitter；本專案由 dataset 先做 jitter，
  再由 policy crop。色彩轉換對每個像素的作用相同，因此順序只會透過 0 與 1 的截斷產
  生影響。
- **不合併 adapter。** 原始實作在每個 checkpoint 都把 LoRA 權重合併進模型
  （`vla-scripts/finetune.py:653-660`）；LeRobot 則分開保存。
- **訓練資料。** `lerobot/libero` 以 LeRobot 格式收錄 LIBERO 的四個 suite。它的示範
  資料是否與原始實作過濾過的 `*_no_noops` RLDS dataset 相同，目前尚未確認。

驗證方式：`tests/test_peft.py` 在小型 policy 上執行 `lerobot-train` 的包裝流程，確
認只有 LoRA adapter、action head 與 proprio projector 可訓練；使用原始的 alpha、
dropout 與初始化方式；adapter 存檔後重新載入結果一致；以及 base policy 能精確載入
VLA 權重，並初始化新增的模組。

#### Smoke test

以 200 步的訓練在實際硬體上驗證完整流程：轉換 base、讀取 `lerobot/libero`、在
`lerobot-train` 中套上 LoRA、儲存 adapter，以及在 `lerobot-eval` 中重新載入。

| 項目                          | 值                                                                                                                                                                                                                                                                                                                |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 指令                          | `sky launch -c openvla-oft-train skypilot/train_lora.yaml --env STEPS=200 --env BATCH_SIZE=2 --env SAVE_FREQ=200 --env LOG_FREQ=10 --env RUN_NAME=smoke`，再執行 `sky exec openvla-oft-train skypilot/libero_eval.yaml --env POLICY_PATH=<checkpoint> --env TASK_IDS="[0]" --env N_EPISODES=2 --env BATCH_SIZE=2` |
| 程式碼                        | `06c414e`                                                                                                                                                                                                                                                                                                         |
| 機器                          | RunPod L40（48 GB）、9 vCPU、125 GB RAM、美國機房，目錄價每小時 $0.82                                                                                                                                                                                                                                             |
| 可訓練參數                    | `lerobot-train` 回報 275,798,023 個（276M）：LoRA 107.9M、action head 151M、proprio projector 17M；論文 Table IV 為 279M，其中包含本專案移除之層上的 LoRA                                                                                                                                                         |
| Loss（正規化 action 上的 L1） | 第 10 步為 1.72，第 120 步之後約為 0.6                                                                                                                                                                                                                                                                            |
| 速度與記憶體                  | Batch size 2 時每步 0.63 秒（每秒 3.2 筆樣本），GPU 記憶體 26.4 GB                                                                                                                                                                                                                                                |
| 存下的 checkpoint             | 801 MB 的 `adapter_model.safetensors`，設定為 `r=32`、`lora_alpha=16`、`lora_dropout=0.0`、`init_lora_weights="gaussian"`、`modules_to_save=["action_head", "proprio_projector"]`，`base_model_name_or_path` 指向轉換後的 base                                                                                    |
| 評估                          | `lerobot-eval` 載入 adapter 與其 base，跑了 2 個 LIBERO-Spatial episode（訓練 200 步後成功率為 0%，符合預期）                                                                                                                                                                                                     |

| 階段                                          | 時間                                              |
| --------------------------------------------- | ------------------------------------------------- |
| 在機器上下載 `openvla/openvla-7b`（15 GB）    | 49 秒                                             |
| 下載 `lerobot/libero`                         | 約 30 秒                                          |
| 建立 policy（建立網路、載入 base、套上 LoRA） | 1 分 36 秒                                        |
| 訓練 200 步                                   | 2 分 21 秒                                        |
| 評估 2 個 episode（含載入 adapter）           | 2 分 53 秒                                        |
| 機器總使用時間                                | 約 15 分鐘，包含一次失敗的嘗試；以目錄價計約 $0.2 |

第一次嘗試失敗，原因是 LeRobot 0.6.1 的 `make_policy` 會把 `dataset_meta` 參數傳給
policy 的建構子；`OpenVLAOFTPolicy` 現在會接受額外的 keyword 參數（`06c414e`）。

### 12. FiLM（OpenVLA-OFT+）

OpenVLA-OFT+ 在 vision backbone 中加入 feature-wise linear modulation（FiLM），讓
policy 更確實地遵循語言指令（論文 §IV-C、App. C）。論文只在 ALOHA 實驗中使用
FiLM：手腕相機讓 policy 容易依賴畫面中的線索而忽略指令；LIBERO 則沒有使用。本專案
預設關閉，以 `--policy.use_film=true` 開啟。

兩個 vision transformer 的每個 block 中，在 attention 與 MLP 子層之間，patch 特徵
`x` 會依下式調變：

```text
x ← (1 + γ) ⊙ x + β,    γ = W_γ c + b_γ,    β = W_β c + b_β
```

其中 `c` 是任務 prompt 的平均語言 embedding，每個 block 各有自己的 `W` 與 `b`。
`γ` 與 `β` 的每個元素對應一個 hidden unit，由同一張影像的所有 patch，以及同一筆樣本
的所有相機影像共用。

| 設定                 | 值                                                        | Config 欄位                        | 來源                                                                                                                                             |
| -------------------- | --------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| 是否啟用             | LIBERO 不使用，ALOHA 使用                                 | `use_film`（False）                | 論文 App. D、Table IV、Table V；原 repo `vla-scripts/finetune.py:83`、`LIBERO.md:106`、`ALOHA.md:66`                                             |
| 調變方式             | `(1 + γ) ⊙ x + β`                                         | （固定）                           | 論文 §IV-C、App. C；原 repo `prismatic/models/film_vit_wrapper.py:72`                                                                            |
| 位置                 | 每個 block 中，attention 子層之後、MLP 之前               | （固定）                           | 論文 §IV-C、Fig. 8；原 repo `prismatic/models/film_vit_wrapper.py:69-75`                                                                         |
| Transformer          | DINOv2 與 SigLIP 都使用                                   | （固定）                           | 論文 App. A（OFT change 6）、App. C；原 repo `prismatic/models/film_vit_wrapper.py:196-198`                                                      |
| 調變單位             | 每個 hidden unit，所有 patch 共用                         | （固定）                           | 論文 §IV-C、App. C；原 repo `prismatic/models/film_vit_wrapper.py:72`                                                                            |
| Projection           | 每個 block 的 `γ` 與 `β` 各一個仿射映射（`4096 → D_ViT`） | （固定）                           | 論文 App. C；原 repo `prismatic/models/film_vit_wrapper.py:53-54`                                                                                |
| 初始化               | PyTorch `nn.Linear` 的預設初始化                          | （固定）                           | 原 repo `prismatic/models/film_vit_wrapper.py:53-54`；論文只提到 `γ` 與 `β` 初始時接近零（App. C）                                               |
| 語言 embedding `c`   | BOS、prompt 與 EOS 在語言模型輸入端 embedding 的平均      | （固定）                           | 論文 §IV-C；原 repo `prismatic/extern/hf/modeling_prismatic.py:575-586`（訓練）、`:994-1003`（推論）、`prismatic/models/film_vit_wrapper.py:242` |
| 平均是否包含 padding | 包含                                                      | `film_mask_padding`（False）       | 原 repo：padding 不是 action 位置，因此 `prismatic/extern/hf/modeling_prismatic.py:581` 會保留它；論文未提及                                     |
| 訓練方式             | 完整訓練，不使用 LoRA                                     | Policy 預設值（`modules_to_save`） | 論文 Table V；原 repo `vla-scripts/finetune.py:854-866`（在 LoRA 之後才加入 FiLM）、`:642-646`（儲存整個 vision backbone）                       |
| 參數量               | 438M                                                      | （推導）                           | 論文 Table V 為 456M，其中包含本專案移除的各 transformer 最後一個 block 的 projection（§2）                                                      |

FiLM 在 LIBERO 上的影響不大：以四個 suite 共同訓練一個 policy，加入 FiLM 的平均成功率
為 97.0%，不加為 96.8%（論文 Table XIV）。

與原始實作的差異（本專案）：

- **合併的 projection。** 原始實作把每個 timm block 包進
  `FiLMedVisionTransformerBlock`，各自帶有 `scale` 與 `shift` 層，因此每個 ViT
  參數都被改名為 `blocks.N.block.*`。本專案的每個 transformer 只有一個
  `FiLMGenerator`，其 `scale` 與 `shift` 層一次產生所有 block 的 `γ` 與 `β`；
  權重的第 `N × D_ViT` 到 `(N + 1) × D_ViT` 列就是第 `N` 個 block 的 projection。
  由於 fan-in 不變，計算結果與初始化分布都和原始實作相同。ViT 的參數名稱維持不變，
  因此 OFT checkpoint 與 LoRA target 都不受影響，PEFT 也能把每個 generator 當成一個
  `modules_to_save` 項目完整訓練。
- **被移除的 block 沒有 projection。** 原始實作也會為最後一個 block 建立
  projection，但該 block 的輸出從未被讀取（§2）。
- **Padding。** 原始實作在訓練時會把 batch 內的 padding 一起平均，推論時（batch
  size 1）則沒有 padding，因此訓練樣本的條件向量會受到同一個 batch 中其他 prompt 的
  影響。`film_mask_padding=True` 只平均 prompt 與 EOS；為了與原始實作一致，預設關閉。

要以 FiLM 進行 fine-tune，在 §11 的 `lerobot-train` 指令加上
`--policy.use_film=true`。Base policy 沒有 FiLM 權重，因此 projection 會從預設初始化
開始，與原始實作相同。

驗證方式：`tests/test_vision_backbone.py` 從 submodule 載入原始的
`film_vit_wrapper.py`，用它包裝一個小型 timm ViT，把合併的 projection 複製到它各
block 的層中，確認兩者產生相同的特徵。`tests/test_model.py` 檢查包含與不包含 padding
時的語言 embedding 平均；`tests/test_peft.py` 檢查 projection 會被完整訓練，且從 base
policy 開始訓練時會被初始化。原作者沒有釋出任何 OpenVLA-OFT+ checkpoint（Hugging Face
Hub 上只有五個 LIBERO 的 OpenVLA-OFT checkpoint），因此端到端的行為尚未與原始實作
對照。

## 引用

本插件是 OpenVLA-OFT 的獨立移植，OpenVLA-OFT 由 Moo Jin Kim、Chelsea Finn 與
Percy Liang 提出。若使用本插件，請引用他們的[論文](https://arxiv.org/abs/2502.19645)：

```bibtex
@article{kim2025fine,
  title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
  author={Kim, Moo Jin and Finn, Chelsea and Liang, Percy},
  journal={arXiv preprint arXiv:2502.19645},
  year={2025}
}
```

## 授權

本 repo 的程式碼採用 [Apache License 2.0](LICENSE) 授權。

- 本插件移植自 [OpenVLA-OFT](https://github.com/moojink/openvla-oft)，其採用 MIT
  授權，Copyright (c) 2025 Moo Jin Kim, Chelsea Finn, Percy Liang。原始程式碼以
  `third_party/openvla-oft` submodule 參照，並未複製到本 repo 中。
- `.pre-commit-config.yaml` 與 `pyproject.toml` 中的工具設定改寫自
  [LeRobot](https://github.com/huggingface/lerobot)，其採用 Apache License 2.0
  授權。
- 本 repo 不含任何模型權重。透過本專案下載或轉換的 checkpoint，例如
  `openvla/openvla-7b` 與原作者釋出的 OpenVLA-OFT checkpoint，適用其各自的授權，
  其中可能包含底層語言模型的 Llama 2 Community License。
