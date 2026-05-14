# 亮度色彩映射 (Luma Color Map)

## 概述

从一张彩色图像出发，转换到 Oklch，并直接使用原始 Oklch 的 Lightness `L0` 作为统一输入轴。当前实现已经稳定拆成 core、API、GUI、CLI 四层：算法通过 core 复用，程序集成通过 API 进入，绘图和编辑器通过 GUI 复用，CLI 只负责交互与调度。

## 分层入口

- Core：`texture_map_toolbox/core/luma.py`
- API：`texture_map_toolbox/api/luma.py`
- GUI 绘图：`texture_map_toolbox/gui/luma_plots.py`
- GUI 编辑：`texture_map_toolbox/gui/qt_editor.py`
- CLI：`texture_map_toolbox/cli/luma.py`

如果调用方没有显式提供图片路径，默认样例图会优先尝试内置候选文件名；若这些文件不存在，则自动回退到 `data/` 目录里第一个可用的支持图片。

## 输入与 mask 来源

输入是一张至少包含 RGB 三通道的彩色图像。推荐直接提供带 alpha 的 PNG；如果输入图没有可用 alpha，也可以额外提供一张与原图同尺寸的二值/灰度 mask。

当前 mask 来源优先级为：

1. 外部 `alpha_mask_path`
2. 可用的嵌入式 alpha
3. 用户点击得到的 connected-region seed mask
4. implicit opaque 继续运行

补充说明：

- PNG 全 1 alpha 会被视作“没有可用 mask”
- `connected-region seed mask` 主要由 Qt 编辑器和目标图对话框触发
- 旧的 border-connected auto-detect helper 仍保留在 core 中做参考、测试和非交互实验，但不再是 Qt 主交互路径

## 处理流程

### 1. 图像加载与预处理

- 读取 RGB 与最终采用的 alpha/mask
- 将图像转为浮点格式 `0-1`
- 构建 `valid_mask` 作为分析区域
- 保留 `alpha_source` 和 `image_warnings`，供 CLI / GUI 展示

### 2. 转换到 Oklch

将 RGB 转换到 Oklch，并直接取原始 Lightness 作为输入轴：

$$
y = L_0
$$

这一步已经完全替代 BT.709 灰度骨架，不再围绕 HSL 或传统灰度桶做主流程。

### 3. 构建样本云与关键点

从所有有效像素中收集样本：

$$
(y_i, C_i, h_i)
$$

关键点位置通过分位数采样生成：

- 先用 quantile 提取 `y_k`
- 保留端点 `0` 和 `1`
- 如果 quantile 结果因重复值退化，则补充均匀抽取的原始样本值，保证关键点数量不低于目标下限

### 4. 拟合连续基础模型

基础模型只拟合：

- `C(y)`
- `h(y)`

hue 不是直接按角度标量插值，而是先转成单位圆向量：

$$
u = \cos(h), \quad v = \sin(h)
$$

然后分别拟合 `u(y)` 和 `v(y)`，最后再用 `atan2` 还原角度。这是当前处理 `0/360` seam 的核心手段。

插值器使用 `PchipInterpolator`，以减少过冲并保持较稳定的形状。

### 5. 用户状态曲线

主流程显式经过用户状态曲线层：

$$
L' = L_t(y), \quad C' = C_t(L'), \quad h' = h_t(L')
$$

默认语义是：

- `Lt(y)` 为 identity
- `Ct(L')` 和 `ht(L')` 默认回退到基础模型

因此在没有外部曲线 JSON 或 GUI override 时，输出会与基础模型保持一致。

### 6. 预览与重建路径

当前同时保留两条执行路径：

#### `original`

- 面向离线高质量生成
- 执行完整重建、gamut compression 和可选评估

#### `fast`

- 面向与 GUI 共用的快速预览路径
- 仍先基于完整有效像素构建基础模型
- 再对输入图做固定比例降采样
- 在 `[0, 1]` 上烘焙一条有限长度的 Oklch LUT
- 在缩图上做 LUT 查表重着色

Qt 编辑器的实时预览与 `fast` 共享同一组 helper。

### 7. gamut compression 与评估

转回 sRGB 时，如果颜色落在 gamut 外，则固定 `L'` 和 `h'`，只压缩 `C'` 直到颜色回到 sRGB gamut。

开启评估时，还会计算：

- PSNR
- CIEDE2000 Delta E 统计

## `valid_mask` 的作用范围

当前 `valid_mask` 只用于：

- 样本云分析
- 基础模型拟合
- 直方图统计
- 评估

但最终 LUT 应用与全分辨率输出会覆盖整张图像，而不是直接把 mask 外区域清零。

## 输出

当前统一输出语义包括：

- 分位数关键点与基础模型
- `Lt / Ct / ht` 状态曲线控制点数量信息
- 重建后的 RGB 图像（可选保存）
- 结果摘要 JSON（可选保存）
- 质量指标（PSNR、Delta E 统计）

## 可视化

当前绘图仍由 matplotlib 承担，生成两组图：

### 对比图（4 宫格）

| 面板 | 内容 |
|------|------|
| 1 | 输入 Oklch Lightness `L0` |
| 2 | 原始图像 |
| 3 | 重建图像 + PSNR |
| 4 | CIEDE2000 误差热力图 |

### 分析图（4 子图）

| 子图 | 内容 |
|------|------|
| 1 | 输入轴 `L0` 的像素分布直方图（对数坐标） |
| 2 | 拟合后的 `C(y)` 曲线 |
| 3 | Oklch LUT 颜色条预览 |
| 4 | 拟合后的 `h(y)` 曲线 |

## Lightness 转换曲线拟合 helper

当前 API 暴露了一个专门针对 `Lt(y)` 的单调拟合 helper：

- `fit_monotonic_lightness_transfer_curve(source_lightness, target_lightness, quantile_count=256)`

它基于经验 CDF / quantile matching 拟合一条单调 Lightness 转换曲线：

$$
L_t(y) \approx F_{target}^{-1}(F_{source}(y))
$$

返回值可以直接作为 `lightness` 控制点写入曲线 JSON。

示例：

```python
from texture_map_toolbox.api.luma import fit_monotonic_lightness_transfer_curve

lightness_points = fit_monotonic_lightness_transfer_curve(
    source_lightness,
    target_lightness,
    quantile_count=256,
)
```

## CLI 调用

推荐通过统一 CLI：

```bash
python -m texture_map_toolbox luma path/to/image.png --algorithm original --curves path/to/curves.json
python -m texture_map_toolbox luma path/to/image.jpg --alpha-mask path/to/mask.png --algorithm original
python -m texture_map_toolbox luma path/to/image.png --algorithm fast --preview-scale 0.25 --preview-lut-size 512
```

常用选项：

- `--algorithm {original,fast}`
- `--alpha-mask path/to/mask.png`
- `--curves path/to/curves.json`
- `--request-json path/to/request.json`
- `--output-image path/to/file.png`
- `--result-json path/to/file.json`
- `--no-plots`
- `--skip-evaluation`
- `--preview-scale`
- `--preview-lut-size`

## 与编辑器的关系

Qt 编辑器不是一套分叉实现，而是直接复用这条主流程的核心语义：

- 相同的输入轴定义 `y = L0`
- 相同的基础模型与 hue wrap-around 处理
- 相同的状态曲线结构
- 相同的快速预览 LUT helper
- 相同的预曲线蓝噪声抖动语义：默认强度为输入图像归一化码值间隔的一半（8bit 为 `0.5/255`），且只扰动 mask 有效区域

更多桌面交互说明见 [docs/hsl_curve_editor_design.md](hsl_curve_editor_design.md)。

### request / result JSON

推荐 GUI 或外部流程通过 JSON 与 CLI 交换数据：

```bash
python -m texture_map_toolbox luma --request-json docs/examples/luma_request.fast.json
```

request JSON 示例：

- [docs/examples/luma_request.original.json](docs/examples/luma_request.original.json)
- [docs/examples/luma_request.fast.json](docs/examples/luma_request.fast.json)

这些示例里的 `image_path` 是占位路径，使用前请替换成你的实际输入图路径。

如果通过 request JSON 调用，也可以额外传入 `alpha_mask_path`。

`dither_strength` 可以省略或写成 `null`，表示按输入图像位深自动取半个码值间隔；写成 `0.0` 表示关闭抖动。

## 外部控制点输入

API / CLI 都支持通过 JSON 文件注入外部控制点：

```bash
python -m texture_map_toolbox luma path/to/image.png --algorithm original --curves path/to/curves.json
```

## 测试覆盖

当前仓库提供一套基于仓库内置样例图的 smoke tests，覆盖：

- core 工作流
- API 调用
- CLI 参数和 request JSON
- matplotlib 绘图 helper
- 编辑器初始化、导出和全分辨率重建
- Lightness 单调分布拟合
- alpha 检查、外部 alpha mask 覆盖和 Qt 警告弹框

运行方式：

```bash
python -m unittest tests.test_luma_smoke tests.test_lightness_transfer_curve tests.test_alpha_input_validation
```

JSON 顶层是一个对象，可包含以下键：

- `lightness`: $L_t(y)$ 的控制点列表
- `chroma`: $C_t(L')$ 的控制点列表
- `hue`: $h_t(L')$ 的控制点列表

每个键的值都应是 `[[x, y], ...]` 形式的二维数组，其中：

- `x` 的范围为 `[0, 1]`
- `lightness` 的 `y` 范围为 `[0, 1]`
- `chroma` 的 `y` 为非负数
- `hue` 的 `y` 以角度表示，单位为度

示例：

```json
{
   "lightness": [[0.0, 0.0], [0.35, 0.30], [0.7, 0.78], [1.0, 1.0]],
   "chroma": [[0.0, 0.0], [0.3, 0.05], [0.6, 0.10], [1.0, 0.02]],
   "hue": [[0.0, 25.0], [0.4, 35.0], [0.7, 180.0], [1.0, 260.0]]
}
```

未提供的曲线会自动回退到默认控制点：

- `lightness` 回退到恒等曲线
- `chroma` 和 `hue` 回退到基础模型关键点

## 依赖

- `numpy`
- `scikit-image`
- `matplotlib`
- `colour-science`
- `scipy`
