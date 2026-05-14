# Oklch 状态曲线编辑器 — 当前实现与设计

> 文件名保留了历史命名 `hsl_curve_editor_design.md`，但当前内容已经完全围绕 Oklch 状态曲线编辑器。

## 功能概述

交互式工具，用于编辑 Oklch 主流程中的三条状态曲线：

- `Lt(y)`：输入轴 `L0` 到输出 Lightness 的映射
- `Ct(L')`：最终 Lightness 到输出 Chroma 的映射
- `ht(L')`：最终 Lightness 到输出 Hue 的映射

当前实现包括两个后端：

- `texture_map_toolbox/gui/editor.py`：历史 matplotlib 版
- `texture_map_toolbox/gui/qt_editor.py`：当前推荐的 Qt 版

推荐入口：

- `python -m texture_map_toolbox`
- `python -m texture_map_toolbox editor --backend qt`

Qt 编辑器与主流程共用同一套 Oklch 语义，默认状态下会回退到基础模型，因此在未手动修改曲线时，输出与默认离线工作流一致。

---

## 当前架构

层级关系为：

- Core：`texture_map_toolbox/core/luma.py`
- API：`texture_map_toolbox/api/luma.py`
- GUI：`texture_map_toolbox/gui/editor.py`
- GUI：`texture_map_toolbox/gui/qt_editor.py`
- CLI：`texture_map_toolbox/cli/editor.py`

### 1. 基础模型

编辑器启动时会先完成一次基础建模：

1. 读取原图和最终采用的 mask
2. 转换到 Oklch
3. 使用原始 Oklch 的 `L0` 作为输入轴
4. 从样本云拟合基础 `C(y)` / `h(y)` 模型

Qt 版本还提供一个启动页，可以在 GUI 内选择：

- 原图路径
- 可选 alpha mask 路径
- 可选初始 curves JSON
- 可选 curves 导出路径

### 2. 控制点与默认模式

编辑器中的三组控制点最终对应 `Lt / Ct / ht`，但 Qt 版本把“显示 handles”与“实际默认基线”分开处理：

- Lightness：默认就是 identity，显示与实际语义一致
- Chroma / Hue：默认只显示稀疏 sampled handles，但默认模式下真正传给 `build_state_curve_set(...)` 的 payload 是 `None`，因此会回退到基础模型的精确基线

当前默认 visible handles 数量是 `STATE_CURVE_CTRL_POINTS`，也就是 16 个左右的稀疏关键点；虚线参考线始终显示精确基线。

### 3. 编辑模式与保存语义

一旦发生以下任一行为，对应曲线就会进入 override 模式：

- 手动拖动控制点
- 调整 Key Points 数量并保留当前曲线值
- 从目标图导入该曲线

Reset to Default 会让曲线回到默认模式。当前保存 JSON 时：

- 只保存已进入 override 模式的曲线
- Reset 后的曲线键会从 JSON 中省略

### 4. 预览路径

实时预览不会在每一帧上跑完整离线主流程，而是走共享的快速路径：

1. 构建 `StateCurveSet`
2. 在 `[0, 1]` 上采样 512 项预览 LUT
3. 对 LUT 执行 chroma-only gamut compression
4. 在缩小预览图上做查表重着色

这条路径和 CLI 的 `--algorithm fast` 共用同一组 helper。

### 5. 全分辨率确认

Qt 版本通过按钮触发一次完整重建，并弹出对比图；matplotlib 版本仍保留旧式快捷键与窗口流。

---

## Qt 版本当前交互

### 1. 曲线面板

每条曲线都包含：

- 一张 plot 背景图
- 一条当前生效曲线
- 一条默认虚线参考曲线
- 一组可拖拽控制点
- Key Points 数量调节 spinbox
- Reset to Default 按钮
- 当前模式标签（默认/已编辑）

其中：

- `Lt(y)` 面板叠加当前输出 Lightness 直方图、原图 Lightness 虚线直方图，以及可选 target 参考直方图
- `Ct(L')` 和 `ht(L')` 面板使用动态色彩背景

### 2. 控制点拖拽

Qt 版当前的拖拽语义是：

- 内部点支持横向和纵向拖动
- 两端端点的 `x` 锁定在 `0` 和 `1`
- 内部点 `x` 不能穿过相邻控制点
- Lightness 会裁剪到 `[0, 1]`
- Chroma 会裁剪到 `>= 0`
- Hue 会保持 canonical 值在 `[0, 360)`

### 3. Key Points 数量控制

当前“增减关键点数量”通过每条曲线自己的 Key Points spinbox 完成：

- 变少：把当前生效曲线重采样到更少的 sparse handles
- 变多：同样按当前生效曲线补采样

这不是“对某个选中点精确插入/删除”的 UI，而是按当前曲线整体重采样。

### 4. Reset to Default

Reset to Default 的语义不是“重采样一遍 sparse handles”，而是：

- 保留当前 visible handle 数量
- 让该曲线重新回到默认模式
- 实际状态曲线重新回退到基础模型基线，不再受 sparse handles 控制

### 5. 导出与全分辨率操作

Qt 版顶部按钮区当前提供三类输出相关操作：

- `Save Curves JSON`：导出当前 override 曲线 JSON
- `Export Image`：执行一次当前状态曲线的全分辨率重建，并直接保存结果图像
- `Full-Resolution Render`：执行一次全分辨率重建，并弹出对比图供检查

### 6. Hue 显示窗口

Hue 曲线本身内部仍存成 canonical 的 `[0, 360)` 角度值，但显示层支持一个“窗口起点”滑块：

- 用户可选择任意 `0-359` 作为显示起点
- 面板显示范围变成 `[start, start + 360]`
- 背景、主曲线、默认虚线和控制点都一起映射到这个显示窗口
- 这只是显示变换，不会改变底层控制点和导出的 JSON

### 7. 目标图导入

Qt 版内置一个目标图对话框，可以一张图分别导入 `L / C / H`：

- `Lt(y)`：读取目标图的 Lightness 样本，并调用 `fit_monotonic_lightness_transfer_curve(...)` 自动生成单调 Lightness 转换曲线，然后重采样到当前 visible point count
- `Ct(L')`：读取目标图并按当前 point count 采样其 `C(y)` 模型
- `ht(L')`：读取目标图并按当前 point count 采样其 `h(y)` 模型

这样既支持“一张目标图同时导入多条曲线”，也支持“多张目标图分别喂给不同曲线”。

### 8. Mask 交互

输入图和目标图在没有可用 mask 时，Qt 路径支持三种处理方式：

- 显式加载外部 mask 文件
- 改用一个或多个 seed 像素生成 connected-region mask
- Continue Without Extra Mask

connected-region seed mask 当前支持：

- 多 seed 点切换
- 颜色误差滑块
- 区域偏移滑块
- 原图 marker overlay
- mask preview 面板

旧的 auto-detect border mask 代码仍保留在 core 中，但不再是 Qt 默认交互。

---

## 性能观察

当前预览主要分成三段：

1. 构建状态曲线并烘焙预览 LUT
2. 在缩小图像上执行 LUT 查表
3. Qt / matplotlib 重绘 UI

结论仍然比较稳定：

- 曲线求值本身不是瓶颈
- LUT 查表已经足够快
- 主要开销仍在 GUI 重绘和事件调度

Qt 版本已经避免了 matplotlib 交互时的一部分绘制成本，但仍是当前桌面 MVP，而不是最终形态。

---

## 当前限制

1. “增加/减少关键点数量”当前是整体重采样，不是对选中点做精确插入/删除。
2. matplotlib 后端没有补齐 Qt 版本的大部分新交互。
3. Hue 显示窗口当前显示的是连续窗口值，而不是循环格式化后的 `0-359` 文本刻度。
4. Qt 版虽然已经可用，但仍然优先保证语义正确和流程统一，而不是追求最终 UI 完整度。

---

## 后续方向

1. 增加显式的 Add Point / Remove Point 交互，而不是只依赖 point-count 重采样。
2. 继续补齐更细的数值输入、批量编辑和更明确的导出/确认 UI。
3. 如有必要，再进一步优化绘制路径或评估更适合实时交互的 GUI 框架。