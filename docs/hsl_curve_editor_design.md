# Oklch 状态曲线编辑器 — 设计与性能分析

## 功能概述

交互式工具，对 Oklch 主流程中的三条状态曲线进行编辑：

- `Lt(y)`：输入轴 `L0` 到输出 Lightness 的映射
- `Ct(L')`：最终 Lightness 到输出 Chroma 的映射
- `ht(L')`：最终 Lightness 到输出 Hue 的映射

**当前实现**：[`scripts/hsl_curve_editor.py`](../scripts/hsl_curve_editor.py)

**推荐 CLI 入口**：`python -m scripts.cli editor ...`

该编辑器已经和 [`scripts/luma_color_map.py`](../scripts/luma_color_map.py) 共用同一套 Oklch 主流程，默认状态下会继承基础模型，因此在未手动改曲线时，和离线主脚本的默认输出一致。

---

## 当前架构

### 1. 基础模型

编辑器启动时先执行一次离线建模：

1. 读取原图并转换到 Oklch
2. 使用原始 Oklch 的 `L0` 作为输入轴
3. 从样本云拟合基础 `C(y)` / `h(y)` 模型

### 2. 控制点语义

编辑器中的三组控制点直接对应最终状态曲线：

- Lightness 曲线：固定数量的均匀 `x` 点
- Chroma 曲线：默认继承基础模型关键点
- Hue 曲线：默认继承基础模型关键点

如果通过 `--curves` 加载外部 JSON，则编辑器会直接使用该文件中的控制点作为初始状态。

### 3. 预览路径

实时预览不直接在每个像素上跑完整的高精度主流程，而是使用一条轻量路径：

1. 根据当前 Lt/Ct/ht 构建 `StateCurveSet`
2. 在 `[0, 1]` 上采样一条 512 项的 Oklch 预览 LUT
3. 对 LUT 执行 chroma-only gamut compression
4. 在缩小预览图上用 `np.take` 做快速查表重着色

这条快速路径现在已经和 CLI 的 `python -m scripts.cli luma --algorithm fast ...` 共用同一组 helper，因此后续 GUI、CLI 和编辑器不会再各自维护一套不同的预览算法实现。

这保留了当前 matplotlib 版编辑器最重要的优化点：

- 缩图预览
- 预分配缓冲区
- LUT 查表而不是逐像素 Python 循环

### 4. 全分辨率确认

用户按 `Enter` 时，编辑器会调用主流程中的全分辨率重建函数，执行一次完整生成，并显示结果对比图。

---

## 当前交互

- 左键拖拽：移动控制点
- 右键点击控制点：重置该点到编辑器初始值
- `S` / `Ctrl+S`：导出当前 Lt/Ct/ht 为 JSON
- `Enter`：执行一次全分辨率重建

导出的 JSON 与 [`scripts/luma_color_map.py`](../scripts/luma_color_map.py) 的 `--curves` 输入格式一致，因此可以直接拿来喂主脚本。

---

## 性能观察

在当前实现里，预览主要分成三段：

1. 构建状态曲线并烘焙 512 项预览 LUT
2. 在缩小图像上执行 LUT 查表
3. 交给 matplotlib 重绘

这轮迁移后的无界面验证中，示例图像的预览阶段大致表现为：

- `state+lut`：约 3 ms
- `recolor`：约 5 ms
- `draw`：主要受 matplotlib 后端影响

结论仍然和旧版一致：

- 曲线求值本身不是瓶颈
- LUT 查表已经足够快
- 主要瓶颈仍然是 matplotlib 的绘制和交互调度

如果后续仍要追求更流畅的 120Hz 拖拽体验，优化方向没有变：

1. 更快的绘制路径，例如 blitting
2. 更快的前端框架，例如 Dear PyGui / PyQt6
3. 仅在必要时再考虑更激进的数值层优化

---

## 当前限制

1. 文件名仍沿用 [`scripts/hsl_curve_editor.py`](../scripts/hsl_curve_editor.py)，但内部语义已完全切到 Oklch。
2. Chroma / Hue 默认会继承基础模型关键点，因此点数较多，编辑精度高但 UI 会更密。
3. 当前没有原生文件对话框，输入图和曲线文件都通过命令行参数传入。
4. 当前没有“控制点数量调整”面板，也没有数值输入框。
5. 当前 GUI 仍是 matplotlib 版，重点是验证算法和工作流，不是最终形态。

---

## 后续方向

1. 增加控制点数量管理和数值输入
2. 把全分辨率确认和导出结果做成更明确的 UI 控件
3. 如果交互流畅度仍不够，再迁移到更适合实时交互的 GUI 框架