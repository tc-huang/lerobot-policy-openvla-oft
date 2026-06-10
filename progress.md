# 進度追蹤（OpenVLA-OFT → LeRobot plugin）

> 計畫全文見 `plan.md`。狀態圖示：⬜ 未開始 / 🟡 進行中 / ✅ 完成 / ⛔ 阻塞。
> 最後更新：2026-06-10（M1 改採兩階段：Phase A 先就地用 vendored `prismatic/`、砍不必要檔；
> Phase B 後續漸進重構。尚未動工）。

## 里程碑總覽

| 里程碑 | 內容 | 狀態 | DoD（完成定義） |
|---|---|---|---|
| M0 | 封裝與環境調和（build-system、修命名 bug、釘相依、editable 安裝） | ⬜ | plugin 被 `importlib.metadata` 看到且 `register_third_party_plugins()` 能 import；確認 installed lerobot 有 `LiberoProcessorStep` |
| M1 | Phase A：就地用 vendored `prismatic/`（砍不必要檔/分支，修 import）；Phase B 後續漸進重構 | ⬜ | CPU 上能載入並對假輸入吐出 `(1, 8, 7)` normalized 動作，`sys.modules` 不含 tensorflow/dlimp/diffusers |
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

### M1 — Phase A：讓 vendored `prismatic/` 的推論路徑可用 ⬜

> 策略已調整：直接沿用已 vendored 的 `prismatic/`，砍掉推論用不到的檔案與分支、修 import，先讓推論能跑；
> 乾淨化（搬進 `model/`、砍死分支）留到 Phase B。詳細與程式碼定位見 `plan.md` 的 M1 段（M1.A0–M1.A4 + Phase B 參考）。

Phase A（本輪）：
- [x] M1.A0 砍推論用不到的子樹：`vla/datasets/`、`training/{strategies,materialize,metrics}`、
      `models/{vlms,vlas,backbones,load,materialize,registry}`、`models/film_vit_wrapper.py`、
      `vla/{materialize,action_tokenizer}`；保留 `extern/hf/*`、`models/{action_heads,projectors}`、
      `vla/constants.py`、`training/train_utils.py`
- [x] M1.A1 修保留檔的絕對 import：`from prismatic.…` → `from lerobot_policy_openvla_oft.prismatic.…`
- [x] M1.A2 去 diffusers：`action_heads.py` 砍 `from diffusers…` 與 diffusion 類別，只留 `MLPResNet` + `L1RegressionActionHead`
- [x] M1.A3 去 `vla/constants.py` 的 import 副作用：`sys.argv` 偵測 + print → 寫死 LIBERO 常數
- [ ] M1.A4 `OpenvlaOftModel` 載入 base VLA（不 trust_remote_code）+ action head + proprio projector + 存 dataset_statistics.json；對外 `predict_action_chunk` 回 `(1, 8, 7)` normalized（batch=1）
- [ ] 釘定可用 transformers 版本並實測 Llama-2 `inputs_embeds` 路徑（先試 fork 4.40.1）
- [ ] DoD：CPU 載入成功且 `sys.modules` 不含 tensorflow/dlimp/diffusers；假輸入吐出 `(1, 8, 7)`

Phase B（M5 gate 後，非阻塞）：
- [ ] 把保留檔逐步搬進乾淨 `model/`、砍 `modeling_prismatic.py` 的 diffusion/FiLM 死分支、移除推論未用檔（藍圖見 `plan.md` M1.0–M1.7）

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
- 數值一致性（影像前處理、prompt/空 token、QUANTILES↔BOUNDS_Q99、平行解碼一致性）。
- 完整 LIBERO eval 需模擬器 + 實務上需 CUDA；Mac 端先做 gate 與小樣本。
- lerobot 0.5.1（installed）vs 0.5.2（reference）版本落差待對齊。

## 變更紀錄
- 2026-06-10：依「LeRobot-native 重寫 + 推論/LIBERO 評測」方向，建立 `plan.md` 與本追蹤表。
- 2026-06-10：細化 M1 為 M1.0–M1.7 子步驟（含程式碼定位、要砍的分支、權重載入流程、DoD）；
  更正「平行解碼有自訂 4D mask」的誤述（實為 `inputs_embeds` + 2D mask、Llama 內部 causal）。
- 2026-06-10：完整上游 `prismatic` 已整包 vendored 進 `src/lerobot_policy_openvla_oft/prismatic/`，
  M1 改採兩階段：**Phase A** 先就地沿用 vendored 套件（砍 `vla/datasets`、`training/strategies`、原生
  載入路徑、FiLM 等不必要檔；修 36 處 `from prismatic.…` 絕對 import；去 `action_heads` 的 diffusers；
  去 `constants.py` 的 `sys.argv`/print 副作用），先讓推論可跑；**Phase B** 通過 M5 gate 後再漸進重構成
  乾淨 `model/`（原 M1.0–M1.7 改為 Phase B 藍圖）。
