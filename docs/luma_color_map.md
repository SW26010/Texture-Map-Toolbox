# 亮度色彩映射 (Luma Color Map)

## 概述

从一张彩色图像出发，先转换到 Oklch，再使用原始 Oklch 的 Lightness ($L_0$) 作为输入轴 $y$。当前实现已经拆成 core、API、GUI、CLI 四层：算法通过 core 复用，外部程序集成通过 API 进入，绘图与编辑器通过 GUI 层复用，CLI 只负责交互与调度。

## 分层入口

- Core：[`texture_map_toolbox/core/luma.py`](../texture_map_toolbox/core/luma.py)
- API：[`texture_map_toolbox/api/luma.py`](../texture_map_toolbox/api/luma.py)
- GUI 绘图：[`texture_map_toolbox/gui/luma_plots.py`](../texture_map_toolbox/gui/luma_plots.py)
- CLI：[`texture_map_toolbox/cli/luma.py`](../texture_map_toolbox/cli/luma.py)

如果调用方没有显式提供图片路径，当前默认样例图会优先尝试内置候选文件名；若这些文件不存在，则自动回退到 `data/` 目录里第一个可用的受支持图片。

## 原理

### 输入

一张至少包含 RGB 三通道的彩色图像。推荐直接提供带 Alpha 的 PNG；如果输入是 JPG，或图像本身没有有效 Alpha，也可以额外提供一个与原图同尺寸的二值 / 灰度 alpha mask 图像。

### 处理流程

1. **图像加载与预处理**
   - 先检查是否提供了外部 `alpha_mask_path`；若有，则它是最高优先级 mask 来源，且必须与原图尺寸一致，允许二值或灰度
   - 如果没有外部 mask，则检查输入图是否带有“可用”的嵌入式 Alpha；PNG 的 alpha 全为 1 会被视作没有可用 mask
   - 若既没有外部 mask，也没有可用嵌入式 alpha，则可以选择自动检测边缘连通的无效背景区域，生成一张同尺寸二值 mask
   - JPG、没有 alpha 的图像、以及 alpha 全为 1 的 PNG 都会发出警告；GUI 会在这种情况下询问是否立即尝试自动检测
   - 提取 RGB 和最终采用的 Alpha，将图像转为浮点格式 (0-1)
   - 使用有效掩码 (`valid_mask`) 标识非透明像素

2. **转换到 Oklch**
   - 将有效像素从 sRGB 转换到 Oklab，再转换到 Oklch
   - 直接取原始 Oklch 的 Lightness 作为输入轴：
     $$y = L_0$$

3. **构建样本云与关键点**
   - 从所有有效像素中收集 $(y_i, C_i, h_i)$ 样本
   - 使用分位数采样提取关键点位置 $y_k$
   - 若分位数采样因重复值退化，则补充均匀抽取的原始样本值，确保关键点数量不低于目标下限

4. **拟合连续曲线**
   - 对每个关键点聚合代表色度 $C(y)$
   - 对 hue 先转为单位圆向量，再聚合和插值，避免 0° / 360° 断裂
   - 使用 `PchipInterpolator` 拟合连续的 $C(y)$、$u(y)$、$v(y)$

5. **用户状态曲线、重建与评估**
    - 当前主流程已显式接入用户状态曲线层：
       $$L' = L_t(y), \quad C' = C_t(L'), \quad h' = h_t(L')$$
    - 当前默认控制点策略为：
       - $L_t(y)$ 为恒等曲线
       - $C_t(L')$ 和 $h_t(L')$ 默认继承基础模型的关键点
    - 因此在未手动修改控制点时，输出仍与基础 $C(y)$ / $h(y)$ 模型一致
   - `valid_mask` 只用于分析、建模、直方图统计和评估；最终 LUT 应用与全分辨率重建会作用到整张图像，而不是把 mask 外区域清零
   - 转回 sRGB 时，如果颜色落在 gamut 外，则固定 $L'$ 和 $h'$，仅压缩 $C'$ 直到回到 sRGB gamut
   - 计算 PSNR（峰值信噪比）评估像素级差异
   - 计算 CIEDE2000 色差（Delta E 2000）评估人眼感知色差
   - 统计 Delta E 的均值、中位数、标准差、最大值、95 百分位数

### 输出

- 分位数关键点和连续的 $C(y)$ / $h(y)$ 拟合模型
- 重建后的 RGB 图像（8-bit 量化）
- 质量指标（PSNR、Delta E 统计值）

## 可视化

脚本会生成两组图表：

**对比图（4 宫格）**
| 面板 | 内容 |
|------|------|
| 1 | 输入 Oklch Lightness ($L_0$) |
| 2 | 原始图像 |
| 3 | 重建图像 + PSNR |
| 4 | CIEDE2000 误差热力图 |

**分析图（4 子图）**
| 子图 | 内容 |
|------|------|
| 1 | 输入轴 $L_0$ 的像素分布直方图（对数坐标） |
| 2 | 拟合后的 $C(y)$ 曲线 |
| 3 | Oklch LUT 颜色条预览 |
| 4 | 拟合后的 $h(y)$ 曲线 |

## 当前实现说明

- 当前数值主链路已全面抛弃 BT.709 灰度和 HSL 分析路径。
- 当前主流程已显式经过 `L_t / C_t / h_t` 状态曲线层；默认控制点为 identity Lightness 加基础模型的 Chroma / Hue 关键点。
- 当前默认关闭预曲线抖动（`DITHER_STRENGTH = 0.0`），但保留了在输入轴 $y$ 上先加抖动再求值的接口。
- 当前在统一接口下同时保留两种算法：
   - `original`: 原始离线高质量主流程
   - `fast`: 与未来 GUI 共用的快速 LUT 预览算法

### Lightness 转换曲线拟合

当前新增了一个专门针对 `L_t(y)` 的单调拟合 helper：

- `fit_monotonic_lightness_transfer_curve(source_lightness, target_lightness, quantile_count=256)`

它接收两组 `Lightness` 浮点样本，数量可以不同，并基于经验 CDF / quantile matching 拟合一条单调转换曲线：

$$L_t(y) \approx F_{target}^{-1}(F_{source}(y))$$

返回值是可直接复用到现有状态曲线体系里的 `[[x, y], ...]` 控制点，也就是 `lightness_control_points`。这比直接拟合多项式更稳，也更符合“让变换后的源分布尽量贴近目标分布”的目标。

示例：

```python
from texture_map_toolbox.api.luma import fit_monotonic_lightness_transfer_curve

lightness_points = fit_monotonic_lightness_transfer_curve(
   source_lightness,
   target_lightness,
   quantile_count=256,
)
```

如果你已经有一份曲线 JSON，也可以把返回值直接写到其中的 `lightness` 字段。

推荐从 API 层直接集成：

```python
from texture_map_toolbox.api.luma import (
   LumaExecutionRequest,
   fit_monotonic_lightness_transfer_curve,
   run_luma_workflow,
)

lightness_points = fit_monotonic_lightness_transfer_curve(
   source_lightness,
   target_lightness,
)

result = run_luma_workflow(
   LumaExecutionRequest(
      image_path="path/to/image.png",
      algorithm="original",
      show_plots=False,
   )
)
```

## CLI 调用

推荐通过统一 CLI 调用：

```bash
python -m texture_map_toolbox luma path/to/image.png --algorithm original --curves path/to/curves.json
python -m texture_map_toolbox luma path/to/image.jpg --alpha-mask path/to/mask.png --algorithm original
```

额外的 CLI 选项：

- `--algorithm {original,fast}`: 在原始离线算法和快速 LUT 算法之间切换
- `--alpha-mask path/to/mask.png`: 用外部同尺寸二值 / 灰度图覆盖输入图自带 alpha
- `--request-json path/to/request.json`: 从统一 request JSON 读取参数
- `--no-plots`: 不弹出 matplotlib 图表
- `--skip-evaluation`: 跳过 PSNR / Delta E 评估，仅保留生成主链路
- `--output-image path/to/file.png`: 保存当前算法输出图像
- `--result-json path/to/file.json`: 输出机器可读的结果 JSON，便于 GUI 或外部流程读取
- `--preview-scale` / `--preview-lut-size`: 控制 `fast` 模式的缩图比例与 LUT 采样数量

### 快速算法

`fast` 模式使用与未来 GUI 预览一致的轻量路径：

1. 仍然基于完整输入图构建 `C(y)` / `h(y)` 基础模型
2. 对输入图做固定比例切片降采样
3. 在 `[0,1]` 上烘焙一条有限长度的 Oklch LUT
4. 在缩图上做 LUT 查表重着色

示例：

```bash
python -m texture_map_toolbox luma path/to/image.png --algorithm fast --preview-scale 0.25 --preview-lut-size 512 --output-image path/to/preview.png
```

该模式的目标是与 GUI 共享预览路径，不替代 `original` 模式的最终高质量生成。

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
