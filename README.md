# Texture-Map-Toolbox

一组围绕 Oklch luma LUT 工作流构建的 Python 工具，当前重点是离线重建与状态曲线编辑。

## 项目结构

```
Texture-Map-Toolbox/
├── README.md                          # 项目说明
├── requirements.txt                   # Python 依赖
├── scripts/
│   ├── cli.py                         # 统一 CLI 入口
│   ├── luma_color_map.py              # 亮度色彩映射脚本
│   └── hsl_curve_editor.py            # Oklch 状态曲线编辑器
├── docs/
│   ├── cli.md                         # CLI 结构与 GUI 准备说明
│   ├── examples/                      # CLI request 示例
│   ├── hsl_curve_editor_design.md     # Oklch 状态曲线编辑器设计说明
│   ├── luma_color_map.md              # 亮度色彩映射详细文档
│   └── oklch_lut_design.md            # Oklch LUT 设计草案
├── luma_color_map.ipynb               # 亮度色彩映射交互式笔记本
└── metallic_smoothness_map.ipynb      # 历史笔记本，当前不作为主工作流
```

## 工具说明

### 1. 亮度色彩映射 (Luma Color Map)

从彩色图像出发，先转换到 Oklch，直接使用原始 Oklch 的 Lightness ($L_0$) 作为输入轴。当前 CLI 在同一接口下同时保留：

- `original`：原始离线高质量主流程
- `fast`：与未来 GUI 共享的快速 LUT 预览算法

- **脚本**: [`scripts/luma_color_map.py`](scripts/luma_color_map.py)
- **笔记本**: [`luma_color_map.ipynb`](luma_color_map.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SW26010/Texture-Map-Toolbox/blob/main/luma_color_map.ipynb)
- **文档**: [`docs/luma_color_map.md`](docs/luma_color_map.md)
- **设计草案**: [`docs/oklch_lut_design.md`](docs/oklch_lut_design.md)

### 2. Oklch 状态曲线编辑器 (Oklch State Curve Editor)

交互式工具，用于编辑 `Lt(y)`、`Ct(L')` 和 `ht(L')` 三条状态曲线。编辑器会在缩小预览图上实时重着色，支持导出 JSON 控制点文件，并可手动触发一次全分辨率重建。

- **脚本**: [`scripts/hsl_curve_editor.py`](scripts/hsl_curve_editor.py)
- **设计说明**: [`docs/hsl_curve_editor_design.md`](docs/hsl_curve_editor_design.md)

## 快速开始

```bash
pip install -r requirements.txt
```

推荐使用统一 CLI：

```bash
# 查看统一命令结构
python -m scripts.cli --help

# 原始高质量算法
python -m scripts.cli luma path/to/image.png --algorithm original

# 使用外部 Lt/Ct/ht 控制点
python -m scripts.cli luma path/to/image.png --algorithm original --curves path/to/curves.json

# 与未来 GUI 共享的快速 LUT 算法
python -m scripts.cli luma path/to/image.png --algorithm fast --preview-scale 0.25 --preview-lut-size 512

# 使用统一 request / result JSON
python -m scripts.cli luma --request-json docs/examples/luma_request.fast.json

# Oklch 状态曲线编辑器
python -m scripts.cli editor path/to/image.png --curves path/to/curves.json
```

兼容旧调用方式：

```bash
python scripts/luma_color_map.py path/to/image.png
python scripts/hsl_curve_editor.py path/to/image.png
```

`docs/examples/*.json` 中的 `image_path` 是占位符，使用前请替换成你的实际输入图路径。

推荐给 GUI 的调用面：

- `run_luma_color_map(..., algorithm="original")`
- `run_luma_color_map(..., algorithm="fast")`
- `launch_editor(...)`

## 依赖

| 包 | 用途 |
|----|------|
| numpy | 数组计算 |
| scikit-image | 图像读写与灰度转换 |
| matplotlib | 可视化 |
| colour-science | 色彩空间转换与色差计算 |
| scipy | PCHIP 插值 |
| Pillow | 图像读写 |
