# 進度追蹤（OpenVLA-OFT → LeRobot plugin）

> 計畫全文見 `plan.md`。狀態圖示：⬜ 未開始 / 🟡 進行中 / ✅ 完成 / ⛔ 阻塞。
> 最後更新：2026-06-10（建立追蹤表，尚未動工）。

## 里程碑總覽

| 里程碑 | 內容 | 狀態 | DoD（完成定義） |
|---|---|---|---|
| M0 | 封裝與環境調和（build-system、修命名 bug、釘相依、editable 安裝） | ⬜ | plugin 被 `importlib.metadata` 看到且 `register_third_party_plugins()` 能 import；確認 installed lerobot 有 `LiberoProcessorStep` |
| M1 | 移植 OFT 模型最小子集（沿用權重，去 prismatic 套件依賴） | ⬜ | CPU 上能載入並對假輸入吐出 `(1, 8, 7)` normalized 動作，不拖進 tensorflow/dlimp |
| M2 | Config 類別 `OpenvlaOftConfig` | ⬜ | `--policy.type=openvla_oft` 可解析；features 與 normalization_mapping 正確 |
| M3 | Processor pipeline + 統計轉換 | ⬜ | preprocessor 的 `pixel_values`/`input_ids`/normalized proprio 與原版 OFT processor 數值吻合 |
| M4 | Policy 類別 `OpenvlaOftPolicy` + `__init__.py` 匯出 | ⬜ | `select_action`/`predict_action_chunk` 可跑；`forward` 暫 raise；類別命名可被 factory 推導 |
| M5 | 正確性 gate（sample pkl 比對 8×7 chunk） | ⬜ | pytest 比對新舊路徑動作 chunk，atol 1e-3 通過 |
| M6 | LIBERO 端到端評測 | ⬜ | `lerobot-eval --env.type=libero --env.task=libero_spatial` ≥50 episodes，成功率與論文同量級 |
| M7 | （可選）轉 LeRobot 格式 checkpoint + 上傳 | ⬜ | `--policy.path=<repo>` 可直接 eval；MDX 記錄可重現指令與成功率 |
| M8 | （盡力）Mac MPS 相容 | ⬜ | MPS/CPU 上能完成 M5 gate；完整 eval 於 GPU 環境 |

## 依賴關係

```
M0 ──┬── M1 ──┐
     └── M2 ──┴── M3 ──┬── M4 ── M5 ── M6 ── (M7)
                       └──────────┘        └── (M8)
```

## 詳細待辦

### M0 — 封裝與環境調和 ⬜
- [x] `pyproject.toml` 補 `[build-system]`（hatchling 或 setuptools）+ src layout 套件設定
- [ ] plugin 相依加 `transformers`（moojink fork 4.40.1 或實測可用版本）、`timm==0.9.x`、`tokenizers`
- [x] 修 `src/lerobot_policy_openvla_oft/__init__.py` 壞掉的舊 stub 匯出
- [x] 對齊 lerobot 版本（0.5.1 → 0.5.2），`uv sync` + `uv pip install -e .`
- [x] 驗證 `register_third_party_plugins()` 能探索到 plugin
- [ ] 確認 installed lerobot 有 `LiberoProcessorStep` 與 8 維 state 行為

### M1 — 移植 OFT 模型最小子集 ⬜
- [ ] 移植 `PrismaticVisionBackbone`（多影像）、`PrismaticProjector`
- [ ] 移植平行解碼 forward / `predict_action`（去掉內建反正規化）
- [ ] 移植 `L1RegressionActionHead`、`ProprioProjector`
- [ ] 從 moojink repo 載入 base VLA + action head + proprio projector + dataset_statistics.json
- [ ] 釘定可用 transformers 版本並實測 Llama-2 平行解碼路徑

### M2 — Config 類別 ⬜
- [ ] `@PreTrainedConfig.register_subclass("openvla_oft")` + `OpenvlaOftConfig`
- [ ] 欄位：`pretrained_oft_repo`、chunk/action 維度、影像/proprio/tokenizer 參數
- [ ] `normalization_mapping`（VISUAL IDENTITY；STATE/ACTION QUANTILES）
- [ ] `validate_features` / delta indices / optimizer/scheduler preset
- [ ] 移除多餘 `raise NotImplementedError`

### M3 — Processor pipeline ⬜
- [ ] 自訂 `OpenvlaOftImageProcessorStep`（resize/center crop/SigLIP+DINOv2 channel stacking）
- [ ] prompt 模板 + `TokenizerProcessorStep`（Llama-2，含空 token 29871）
- [ ] Rename / AddBatchDim / Device / Normalizer 串接（pre）
- [ ] Unnormalizer + Device(cpu)（post）
- [ ] OFT `dataset_statistics.json` → LeRobot `dataset_stats` 轉換工具
- [ ] 逐項比對 preprocessor 輸出與原版 OFT processor

### M4 — Policy 類別 ⬜
- [ ] `OpenvlaOftPolicy`（`config_class` / `name="openvla_oft"`）
- [ ] `__init__` 組裝載入模型 + `reset()` 建 action queue
- [ ] `predict_action_chunk` 回 `(B, 8, 7)` normalized
- [ ] `select_action` 照抄 ACT/pi0 queue 範式
- [ ] `forward` 暫 raise；`get_optim_params`
- [ ] `__init__.py` 匯出三者

### M5 — 正確性 gate ⬜
- [ ] 用 `sample_libero_spatial_observation.pkl` 比對新舊路徑 8×7 chunk（atol 1e-3）
- [ ] 放進 `tests/`（pytest）

### M6 — LIBERO 評測 ⬜
- [ ] `lerobot-eval --policy.type=openvla_oft --env.type=libero --env.task=libero_spatial --eval.n_episodes=50`
- [ ] 成功率與論文（約 97%）對比；落差大則回 M5 比對中間張量

### M7 — 轉檔上傳（可選）⬜
- [ ] `convert` 腳本產出 LeRobot 格式 checkpoint + 內嵌統計的 processors
- [ ] `--policy.path=<repo>` 驗證；policy MDX 記錄結果

### M8 — MPS 相容（盡力）⬜
- [ ] attention 改 sdpa/eager、device 串接、dtype 處理
- [ ] MPS/CPU 完成 M5 gate；完整 eval 於 GPU 環境

## 風險與待決事項
- transformers 版本相容性（移植的 Llama-2 平行解碼路徑對版本敏感，stock 5.x 不相容）。
- 數值一致性（影像前處理、prompt/空 token、QUANTILES↔BOUNDS_Q99、平行解碼 mask）。
- 完整 LIBERO eval 需模擬器 + 實務上需 CUDA；Mac 端先做 gate 與小樣本。
- lerobot 0.5.1（installed）vs 0.5.2（reference）版本落差待對齊。

## 變更紀錄
- 2026-06-10：依「LeRobot-native 重寫 + 推論/LIBERO 評測」方向，建立 `plan.md` 與本追蹤表。
