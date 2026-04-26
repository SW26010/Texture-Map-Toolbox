# Texture-Map-Toolbox

一组用于 3D 纹理贴图处理的 Python 工具，包含亮度色彩映射和 PBR 材质贴图生成两个工具。

## 项目结构

```
Texture-Map-Toolbox/
├── README.md                          # 项目说明
├── requirements.txt                   # Python 依赖
├── scripts/
│   ├── luma_color_map.py              # 亮度色彩映射脚本
│   ├── hsl_curve_editor.py            # Oklch 状态曲线编辑器
│   └── metallic_smoothness_map.py     # 金属度/光滑度贴图生成脚本
├── docs/
│   ├── hsl_curve_editor_design.md     # Oklch 状态曲线编辑器设计说明
│   ├── luma_color_map.md              # 亮度色彩映射详细文档
│   ├── oklch_lut_design.md            # Oklch LUT 设计草案
│   └── metallic_smoothness_map.md     # 金属度/光滑度贴图详细文档
├── luma_color_map.ipynb               # 亮度色彩映射交互式笔记本
└── metallic_smoothness_map.ipynb      # 金属度/光滑度贴图交互式笔记本
```

## 工具说明

### 1. 亮度色彩映射 (Luma Color Map)

从彩色图像出发，先转换到 Oklch，直接使用原始 Oklch 的 Lightness ($L_0$) 作为输入轴。脚本基于 per-pixel 样本云拟合连续的 $C(y)$ / $h(y)$ 曲线，用于重建图像并通过 PSNR 和 CIEDE2000 评估质量。

- **脚本**: [`scripts/luma_color_map.py`](scripts/luma_color_map.py)
- **笔记本**: [`luma_color_map.ipynb`](luma_color_map.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SW26010/Texture-Map-Toolbox/blob/main/luma_color_map.ipynb)
- **文档**: [`docs/luma_color_map.md`](docs/luma_color_map.md)
- **设计草案**: [`docs/oklch_lut_design.md`](docs/oklch_lut_design.md)

### 2. Oklch 状态曲线编辑器 (Oklch State Curve Editor)

交互式工具，用于编辑 `Lt(y)`、`Ct(L')` 和 `ht(L')` 三条状态曲线。编辑器会在缩小预览图上实时重着色，支持导出 JSON 控制点文件，并可手动触发一次全分辨率重建。

- **脚本**: [`scripts/hsl_curve_editor.py`](scripts/hsl_curve_editor.py)
- **设计说明**: [`docs/hsl_curve_editor_design.md`](docs/hsl_curve_editor_design.md)

### 3. 金属度/光滑度贴图生成 (Metallic Smoothness Map)

从 RGB 通道用作蒙版的输入图像出发，按通道阈值分割材质区域，为每个区域指定金属度、光滑度和颜色，输出两张 PBR 贴图。

- **脚本**: [`scripts/metallic_smoothness_map.py`](scripts/metallic_smoothness_map.py)
- **笔记本**: [`metallic_smoothness_map.ipynb`](metallic_smoothness_map.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SW26010/Texture-Map-Toolbox/blob/main/metallic_smoothness_map.ipynb)
- **文档**: [`docs/metallic_smoothness_map.md`](docs/metallic_smoothness_map.md)

## 快速开始

```bash
pip install -r requirements.txt
```

```bash
# 亮度色彩映射
python scripts/luma_color_map.py

# 使用外部 Lt/Ct/ht 控制点
python scripts/luma_color_map.py path/to/image.png --curves path/to/curves.json

# Oklch 状态曲线编辑器
python scripts/hsl_curve_editor.py path/to/image.png --curves path/to/curves.json

# 金属度/光滑度贴图生成
python scripts/metallic_smoothness_map.py
```

## 依赖

| 包 | 用途 |
|----|------|
| numpy | 数组计算 |
| scikit-image | 图像读写与灰度转换 |
| matplotlib | 可视化 |
| colour-science | 色彩空间转换与色差计算 |
| scipy | PCHIP 插值 |
| Pillow | 图像读写 |
