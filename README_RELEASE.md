# SVDB — 变化去模糊发布包运行说明

本文件是发布包的运行说明。**图形入口是变化去模糊窗口**；命令行只用于准备
输入、批量运行和发布校验。

- 窗口入口：`scripts/start_ui.sh` → 浏览器 `http://127.0.0.1:8765/`
- 批量/校验：`scripts/build.sh`
- 契约、核场、参数与输出目录的约定：见下文「输入契约」

---

## 1. 锁定环境

发布包锁定数值依赖与线程数，保证同一输入得到同一结果。

| 项 | 锁定值 |
| --- | --- |
| Python | CPython 3.11 |
| numpy | 1.26.4 |
| scipy | 1.13.1 |
| Pillow | 10.4.0 |
| pytest | 8.3.3 |
| 线程 | 每类后端 1 个 worker |

安装精确版本：

```bash
python3 -m pip install -r requirements-lock.txt
```

线程由 `svdeblur/threads.py` 和 `scripts/env.sh` 双重设置：
`OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS /
NUMEXPR_NUM_THREADS / VECLIB_MAXIMUM_THREADS / BLIS_NUM_THREADS = 1`。
契约里若声明 `"threads"` 不为 1，加载会以 `stage=threads` 明确失败，
而不是悄悄按别的线程数运行。

---

## 2. 图形入口：变化去模糊窗口

```bash
scripts/start_ui.sh           # 默认 127.0.0.1:8765
SVDB_HOST=0.0.0.0 SVDB_PORT=9000 scripts/start_ui.sh
```

脚本先打印 numpy/scipy/Pillow 版本，缺依赖会直接退出，不会打开一个坏窗口。

窗口内操作顺序：

1. **DEGRADED SAMPLE** — 选择内置样本并 `Load sample`。样本都是从清晰图真实
   生成的退化图，参数与随机种子固定。
2. **KERNEL FIELD EDITOR** — 点选核格，用 centre/spread 滑块编辑 PSF，
   `Write PSF` 写入；`Mark missing` 移除核格；`Fill holes` / `Fill all from
   selected` / `Mirror cell` 补全核场；`Apply grid/support` 改变网格与支撑。
   在任一图像上右键可探测该点的插值核。
3. **Solver** — 设置 σ、迭代数、TV 权重 λ、TV ε、damping μ、noise-floor τ、
   边界与 signed-kernel 策略，按 `Start restoration`。
4. **结果区** — 四宫格为 observed / restored / local kernel / **residual**；
   residual 面板是服务端实际计算的 `A(restored) − observed`（前向残差），
   不是两张 PNG 的客户端相减。下方胶片条是中间迭代，右侧是 PSNR、残差 RMS、
   停止理由、三条迭代曲线，以及当前**迭代残差** `‖r‖` / `‖r‖/‖b‖`。
5. **由变化去模糊界面提示异常** — `MISSING_KERNEL`、`NEGATIVE_WEIGHT`、
   `SUPPORT_TOO_LARGE` 等诊断即时显示；存在 error 级诊断时恢复被拒绝，
   原因同时写入结果区红色错误面板与诊断列表，不会生成恢复图。

导出的图（restored / residual / kernel sheet / curves / report.json）都带同一
`run_id`，来自触发导出的同一组参数。

---

## 3. 输入契约

一个任务是一个 JSON 文件（示例见 `release/cases/*/task.json`），包含四类输入：

```json
{
  "contract_version": 1,
  "name": "gradient",
  "observed_image": "release/cases/gradient/observed.png",
  "kernel_field":    "release/cases/gradient/field.json",
  "output_dir":      "out/release/gradient",
  "reference_image": "release/cases/gradient/reference.png",
  "boundary": "reflect",
  "regularization": {
    "iters": 300, "sigma": 0.004, "tv_eps": 0.008,
    "lam": null, "lam_rel": 0.02, "damping_mu": 0.0,
    "tol": 1e-09, "noise_floor_factor": 0.8
  },
  "memory_budget_mb": 512,
  "threads": 1,
  "allow_signed": false
}
```

各字段含义与强制规则：

| 输入 | 规则 | 失败 stage / code |
| --- | --- | --- |
| `observed_image` | **必须**是单通道 8-bit（PIL mode `L`）；2-D；边长 16–4096 | `image` / `UNSUPPORTED_IMAGE_DIMENSION`、`UNSUPPORTED_IMAGE_DTYPE`、`UNSUPPORTED_IMAGE_SIZE` |
| `kernel_field` | JSON 必须带 `field_version: 2`；不得有缺失/非有限/零和核格；支撑必须为奇数且不超图像 | `kernel_field` / `KERNEL_FIELD_VERSION`、`MISSING_KERNEL`、`KERNEL_FIELD_MISSING` |
| `regularization` | `iters≥1`、`sigma>0`、`tv_eps>0`、`lam≥0` | `regularization` / `BAD_REGULARIZATION` |
| `output_dir` | 必须可创建、可写 | `output` / `OUTPUT_READONLY` |
| `memory_budget_mb` | 预估峰值工作集必须在预算内（见 §5） | `memory_budget` / `MEMORY_BUDGET_EXCEEDED` |
| `threads` | 必须为 1 | `threads` / `UNSUPPORTED_THREADS` |
| `reference_image`（可选） | 与 observed 同尺寸，用于 PSNR | `image` / `REFERENCE_SHAPE_MISMATCH` |

彩色（RGB/RGBA）、16-bit、调色板或 3-D 数组都会被**明确拒绝**，并在消息里
说明应显式 `convert("L")`。缺核文件或核场版本不符同样明确失败——不会静默补核或
沿用旧格式。

`field.json` 由核场编辑器导出，结构：

```json
{ "field_version": 2, "grid_shape": [3,3], "support": [11,11],
  "interpolation": "bilinear", "normalization": "unit_sum",
  "cells": [[[[...]]]] }
```

---

## 4. 启动自检（不是“能 import”）

```bash
scripts/build.sh build        # 或
python3 -m svdeblur.cli selfcheck --out out/selfcheck
```

自检会**实际执行**一次 48×48、2×2 网格、7×7 支撑的小型恢复，然后：

1. 独立重建算子，从**已写出的 restored.png** 重新计算前向残差 `‖A x̂ − b‖`；
2. 校验输出尺寸等于输入尺寸、像素有限；
3. 对比独立残差与求解器记录的最终残差（相对差 < 1e-9）；
4. 校验 restored/residual/kernels/curves/report 全部存在且可解析。

任一阶段失败会返回 `passed=false` 与 `failed_stage`，命令行退出非零。

---

## 5. 内存预算与超限处理

任务声明 `memory_budget_mb`。加载时按

```
峰值 ≈ 8 字节 × H×W × (活动核格 + 10 + 快照数) × fft_factor(3)
```

估算工作集（活动权重图 + 求解器常驻数组 + 快照 + FFT 临时缓冲）。超限时在
任何计算开始前以 `stage=memory_budget` 失败，并给出降预算建议（减小网格/支撑、
图像或快照数，或显式提高预算）。求解过程中真正的 `MemoryError` 也会被转换为
`MEMORY_ERROR` 失败记录。

---

## 6. 取消与失败语义

- 恢复图、残差图、曲线等**只在求解完成后**原子写出（临时文件 + `os.replace`）。
- 取消（`--cancel-file` 出现或收到 SIGTERM/SIGINT）时，运行记录
  `status=cancelled`、保留失败阶段，并**删除 restored.png / residual.png**，
  旧图也一并清除。被取消的目录里不会留下可被误认为完成的恢复图。
- 失败任务写 `run.json` 与 `report.json`，包含 `stage`、`code`、`message`
  以及**实际采用的参数**（`params` / `declared_params`）。

批量运行的退出码：任一真实失败或任一“预期失败未按要求发生”都会让
`scripts/build.sh` 以非零退出。

已内置的“干净目录”测试：

```bash
python3 scripts/test_clean_dir.py
```

它在临时目录中验证三件事（全部通过才返回 0）：

| 场景 | 期望 |
| --- | --- |
| 核场版本不匹配（`field_version: 1`） | 在 `kernel_field` 阶段失败，无 restored.png |
| 输出目录只读 | 在 `output` 阶段失败，无 restored.png |
| 运行中取消（取消标记 / SIGTERM） | `status=cancelled`，无 restored.png |

---

## 7. 发布验证：数值容差

`scripts/build.sh` 会连续跑两遍同一清单，并逐案例比较：

```bash
python3 -m svdeblur.cli verify \
  --out out/release/gradient --reference out/release_ref/gradient --threads 1
```

允许误差及来源（写死在 `svdeblur/verify.py`，不是事后调参）：

| 指标 | 允许值 | 来源 |
| --- | --- | --- |
| `float_delta`（float64 恢复图最大绝对差） | 1e-10 | 两次运行代码路径相同、单线程；仅 BLAS/FFT 归约顺序可能不同 |
| `png_delta`（uint8 量化后最大差） | 0 | 上述浮点差 < 1e-7 个灰阶 |
| `residual_relative_delta` | 1e-9 | 独立重算前向残差 RMS 的相对差 |

验证器同时要求 `restored.png` 与 `restored.npz` 的尺寸等于输入尺寸；**把验证图
缩小以掩盖不稳定会被判为失败**（报告 `problems` 里会写明 downsampled）。

本机实测：4 个案例 `float_delta=0.0`、`png_delta=0`、残差相对差 `0.0`。

---

## 8. 批量运行

```bash
python3 -m svdeblur.release release/manifest.json --jobs 2
python3 -m svdeblur.release release/fixtures/manifest.json --jobs 2
```

- 每个案例在独立子进程中运行，继承锁定的线程环境，互不影响。
- 清单条目可带 `"expect_failure": "<stage>"`，只有在该阶段失败才记为
  `expectation_met=true`。
- 输出：每个案例目录下 `restored.png`、`residual.png`、`kernels.png`、
  `curves.png`、`report.json`、`run.json`。

单案例运行：

```bash
python3 -m svdeblur.cli run release/cases/gradient/task.json --save-npy
```

`--save-npy` 额外写 `restored.npz`（浮点数组），供容差比较使用。

---

## 9. 测试

```bash
python3 -m pytest tests -q          # 数值 + 发布契约/自检/取消/UI 载荷
python3 scripts/test_clean_dir.py   # 干净目录：版本不符 / 只读 / 中止
python3 scripts/ui_check.py         # 真实浏览器打开窗口并操作（需先启动服务）
```

`scripts/ui_check.py` 用 Playwright 实际打开窗口：编辑核场 → 开始恢复 →
核验恢复图与**服务端前向残差图**不是同一张、迭代残差与 API 一致 →
标记缺核 → 确认界面提示并拒绝恢复 → 补核 → 再次恢复成功。

启动服务后运行：

```bash
scripts/start_ui.sh &
SVDB_URL=http://127.0.0.1:8765 python3 scripts/ui_check.py
```

（在缺少浏览器系统库的容器里，UI 检查会说明无法运行；此时至少运行
`pytest` 与 `scripts/test_clean_dir.py`，并把 UI 检查标记为未执行而非通过。）

---

## 10. 产物清单

| 路径 | 内容 |
| --- | --- |
| `release/cases/<name>/` | observed / reference / field.json / task.json |
| `release/manifest.json` | 4 个发布案例的批量清单 |
| `release/fixtures/` | 4 个“预期失败”契约及其清单 |
| `out/release/<name>/` | restored / residual / kernels / curves / report.json / run.json |
| `out/fixtures/<name>/` | 预期失败的记录（含 stage 与参数） |
| `out/selfcheck.json` | 启动自检结果 |
| `out/verify_<case>.json` | 容差验证结果 |
| `out/ui_check/` | 浏览器检查截图 |

所有产物在固定输入与锁定线程下可复现；`run_id` 每次运行不同，但图像与数值相同。
