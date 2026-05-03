# Oklch LUT 方案设计与落地状态

这份文档不再把 Oklch LUT 当成纯草案，而是记录“哪些设计已经落地，哪些仍然是后续方向”。当前主线已经从 notebook 试验期进入统一 core/API/GUI/CLI 结构。

## 当前目标

从彩色贴图的像素分布中提取一个“由亮度驱动的色彩变化模型”，并同时满足两类场景：

1. 离线高质量重建与评估
2. 可交互编辑、可快速预览的 LUT 工作流

## 已经落地的设计决策

### 1. 输入轴统一为原始 Oklch Lightness

系统不再围绕 BT.709 灰度或 HSL Lightness 建模，而是统一使用：

$$
y = L_0
$$

这意味着所有基础模型和状态曲线都围绕真正的 Oklch 亮度工作。

### 2. 基础模型只拟合 `C(y)` 和 `h(y)`

当前基础建模流程从有效像素中拟合：

- `C(y)`：亮度到 Chroma 的关系
- `h(y)`：亮度到 Hue 的关系

关键点由 quantile 驱动生成，插值使用 `PchipInterpolator`。

### 3. Hue 已经采用 seam-safe 的单位圆表示

Hue 不再直接按角度标量插值，而是转成：

$$
u = \cos(h), \quad v = \sin(h)
$$

分别做聚合、拟合和求值，再通过 `atan2` 还原角度。这一设计已经真实落在 core 中，并且被 Qt 编辑器和 CLI 共用。

### 4. 状态曲线层已经接入主流程

当前状态曲线定义为：

$$
L' = L_t(y), \quad C' = C_t(L'), \quad h' = h_t(L')
$$

这三条曲线不再只是 GUI 想法，而是已经成为 `original` 和 `fast` 两条路径的统一语义。

### 5. 快速预览路径已经落地

旧草案里曾把“快速预览路径”写成未来计划。当前这部分已经实现，并且通过统一接口暴露为：

- CLI：`--algorithm fast`
- Qt 编辑器：实时预览

共享路径的大致流程是：

1. 用完整有效像素建基础模型
2. 构建当前 `StateCurveSet`
3. 在 `[0, 1]` 上烘焙有限长度 LUT
4. 在缩小预览图上查表重着色

它不是一套与主流程割裂的“另一个算法”，而是同一语义下的轻量执行路径。

### 6. 桌面编辑器已经落地到 Qt

当前推荐桌面入口已经是 Qt 编辑器，具备：

- 启动页
- 原图/预览图并排显示
- `Lt / Ct / ht` 三条曲线面板
- 稀疏关键点编辑
- 目标图导入
- connected-region seed mask
- Hue 显示窗口滑块

因此文档中所有“未来应实现实时预览”“未来应支持 GUI 文件选择”之类表述都已经过时。

## 当前数据与执行语义

### 1. mask 语义

当前主路径中的 mask 优先级为：

1. 外部 mask
2. 可用嵌入式 alpha
3. connected-region seed mask
4. implicit opaque

旧的 auto-detect border mask 代码仍保留，但已经不再是 Qt 主交互逻辑。

### 2. gamut 处理

当前转回 sRGB 时采用 chroma-only gamut compression：

- 固定 `L'`
- 固定 `h'`
- 只压缩 `C'` 直到颜色回到 sRGB gamut

### 3. 默认模式与 sparse handles

Qt 编辑器把“默认精确基线”和“可编辑 sparse handles”分离：

- Chroma / Hue 默认显示少量 handles
- 但默认状态下真正生效的曲线仍回退到基础模型精确基线
- 只有进入 override 模式后，才会把当前控制点写入状态曲线和 JSON

这项设计已经解决了“基础模型关键点太密导致 GUI 不可操作”的问题。

## 与旧设计稿的差异

以下旧说法已经不再准确：

- “实时预览路径尚未实现”
- “GUI 后续再考虑与 fast 路径共享”
- “scripts/ 下某些入口是当前主执行面”

当前真实入口已经是：

- `texture_map_toolbox/core/luma.py`
- `texture_map_toolbox/api/luma.py`
- `texture_map_toolbox/cli/luma.py`
- `texture_map_toolbox/gui/qt_editor.py`

## 仍然是后续方向的部分

1. 更细粒度的 Add Point / Remove Point，而不是只靠 point-count 重采样。
2. 更丰富的数值输入、批量编辑与曲线约束 UI。
3. 更系统的“预览误差 vs 最终图误差”量化方式。
4. 如果后续扩展更多纹理类别，是否继续复用同一套 `L0 -> Oklch -> StateCurveSet` 主线。

## 参考实现位置

- 数值核心：`texture_map_toolbox/core/luma.py`
- API：`texture_map_toolbox/api/luma.py`
- CLI：`texture_map_toolbox/cli/luma.py`
- Qt 编辑器：`texture_map_toolbox/gui/qt_editor.py`
- matplotlib 编辑器：`texture_map_toolbox/gui/editor.py`