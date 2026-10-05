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
| 輸入影像              | 1 張第三人稱 + 1 張手腕相機   | Dataset features；vision backbone         |
| 機器人狀態輸入        | 是                            | Dataset features；proprio projector       |
| 輸入影像尺寸          | 224 × 224                     | Vision backbone                           |
| LoRA rank             | 32                            | LoRA fine-tuning                          |
| 影像增強              | 90% random crop、color jitter | Processor 與訓練                          |
| FiLM                  | 否                            | 不在範圍內                                |
