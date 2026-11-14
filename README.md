# 空间变化模糊的非盲恢复 / Spatially Varying Deblurring

输入是退化图 `b` 与已知的位置相关模糊核场 `k(y,x)`。本仓库用**显式的双线性
（tent）核插值**构造前向算子 `A` 与它的**精确伴随** `Aᵀ`，在噪声模型与约束下用
FISTA 反演，并把数据残差、停止理由和中间迭代全部落盘/上屏。

> **发布包说明见 [`README_RELEASE.md`](README_RELEASE.md)**：锁定依赖与线程、
> 输入契约、启动自检、内存预算、取消语义、干净目录测试与数值容差验证。

## 图形入口：变化去模糊窗口

发布包的主入口是**变化去模糊（change-aware deblurring）窗口**。界面是纯标准库
HTTP 服务 + 静态页面，由**变化去模糊内核**驱动：

```bash
scripts/start_ui.sh                  # 默认 http://127.0.0.1:8765/
# 等价于（先设置锁定线程）：
python3 -m svdeblur.server --host 0.0.0.0 --port 9000
```

然后浏览器打开 `http://127.0.0.1:8765/`。终端会打印一行
`SVDB restoration window: ...`，Ctrl-C 停止。启动脚本会先校验数值依赖。

界面使用流程：

1. **左侧「DEGRADED SAMPLE」** 选择内置样本（这些样本都是从清晰图实际生成的退化图，
   参数固定、种子固定），点「Load sample」。
   `diag_missing` / `diag_negative` / `diag_oversized` 是三个诊断样本。
2. **「KERNEL FIELD EDITOR」** 点击任意核格选中它，用 centre/spread 滑块改它的
   PSF；右键（或长按）在任一图像上探测该点的插值核，右侧面板会显示四个贡献格与
   权重的加权结果。改「grid rows/cols」「support」后点「Apply grid/support」。
   「Fill holes」用于补上缺失的核。
3. **诊断**：每次编辑后会立刻调用 `/api/diagnostics`，`MISSING_KERNEL`、
   `NEGATIVE_WEIGHT`、`SUPPORT_TOO_LARGE` 等会显示在左栏底部。存在 error 级诊断时
   「Start restoration」直接被拒，反馈会写回结果区红色错误面板。
4. **「Start restoration」**：结果区四宫格显示 observed / restored / local kernel /
   **residual**；residual 是服务端对本次恢复图实际计算的前向残差
   `A(restored) − observed`（不是两张 PNG 的客户端相减）。下方胶片条是中间迭代
   结果，点任意一帧可在 restored 面板查看；右侧是 PSNR、残差 RMS、停止理由、
   三条迭代曲线与当前**迭代残差**。
5. **「Run & export all artifacts」**：全部导出（恢复图、残差图、核场图、曲线、报告
   JSON）都带同一个 `run_id`，来自触发导出时的同一组参数。

## 命令行（仅辅助）

命令行不替代窗口，只用于批量与校验：`python3 -m svdeblur.cli
{run,selfcheck,verify,export-field}` 与 `python3 -m svdeblur.release <manifest>`。
详见 `README_RELEASE.md`。

## 算子与求解

- `svdeblur/field.py` — 核场：`(gh,gw,kh,kw)` 控制格 + 缺失掩码；`_tent` 给出
  非负、逐像素和为 1 的双线性权重；`interpolate_at` 与 `weight_maps` 使用同一规则。
- `svdeblur/operator.py` — `A x = Σ_c m_c ⊙ (k_c ⊛ E x)`，其中 `m_c` 是 tent 权重
  图、`E` 是边界延拓。伴随 `Aᵀ y = Eᵀ Σ_c k_cᵀ ⊛ (m_c ⊙ y)`，`E` 的索引折叠精确
  通过 `np.add.at` 实现。边界模式：`zero / reflect / symmetric / replicate`。
- `svdeblur/solve.py` — FISTA + Armijo 回溯；数据项 `0.5‖Ax−b‖²`，正则项为平滑
  TV；噪声地板 `τ·σ√N`，残差触底反弹的半收敛保护；返回 `residuals / residual_abs /
  objectives / tv_terms / snapshots / stop_reason / stop_detail`。
- `svdeblur/synth.py` — 从清晰纹理真正卷积生成退化样本（核场完整时生成，再根据
  需要移除缺失格）。
- `svdeblur/pipeline.py` — 生成数据、基准、图像与 sha256 manifest。
- `svdeblur/server.py` — UI 的 JSON 后端。

## 数值验证

```bash
python3 -m pytest tests -q
```

`tests/test_core.py` 断言：

| 断言 | 内容 |
| --- | --- |
| 插值 | tent 权重非负且逐像素和为 1；点插值等于权重图组合 |
| 伴随 | 四种边界下 `⟨Ax,y⟩ = ⟨x,Aᵀy⟩`（相对误差 <1e-10）；`E`/`Eᵀ` 折叠配对 |
| 局部核等价 | 冲激响应逐像素等于该点插值核；均匀核场退化为单卷积 |
| 归一 | 单位和的核；未归一/零和/负权诊断分级 |
| 边缘 | 延拓后边缘残差与内部同为机器精度，无边缘伪影环 |
| 求解 | PSNR 提升、残差下降、停止理由被记录 |
| 不放大噪声 | 迭代预算 150/400/900 都在同一迭代停下、输出一致 |
| 可复现 | 同参数同种子生成完全相同的样本 |

`tests/test_release.py` 覆盖发布层：启动自检真实执行并核验残差与尺寸、
缺核/彩色图像/版本不符/内存超限的阶段化失败、取消后不留下恢复图、
以及 UI 恢复载荷里残差是服务端实际前向计算。

## 复跑像素结果

```bash
python3 -m svdeblur.pipeline --out out --data data --diagnostics
```

会重新生成 `data/` 与 `out/` 下的像素产物，`out/manifest.json` 记录每个文件的
sha256 和完整参数；相同参数下 manifest 哈希稳定。

## 依赖

`numpy`、`scipy`（FFT 卷积）、`Pillow`（PNG）。UI 服务与页面不需要额外包。
运行测试需要 `pytest`。发布包精确版本见 `requirements-lock.txt`。
