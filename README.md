# Texture-Map-Toolbox

一组用于 3D 纹理贴图处理的 Python 工具，包含亮度色彩映射和 PBR 材质贴图生成两个工具。

## 项目结构

```
Texture-Map-Toolbox/
├── README.md                          # 项目说明
├── requirements.txt                   # Python 依赖
├── scripts/
│   ├── luma_color_map.py              # 亮度色彩映射脚本
│   └── metallic_smoothness_map.py     # 金属度/光滑度贴图生成脚本
├── docs/
│   ├── luma_color_map.md              # 亮度色彩映射详细文档
│   └── metallic_smoothness_map.md     # 金属度/光滑度贴图详细文档
├── luma_color_map.ipynb               # 亮度色彩映射交互式笔记本
└── metallic_smoothness_map.ipynb      # 金属度/光滑度贴图交互式笔记本
```

## 工具说明

### 1. 亮度色彩映射 (Luma Color Map)

从彩色图像出发，按 ITU-R BT.709 标准计算亮度，为每个灰度值 (0–255) 建立对应的平均 RGB 颜色查找表 (LUT)。支持用 LUT 重建图像并通过 PSNR 和 CIEDE2000 评估质量。

- **脚本**: [`scripts/luma_color_map.py`](scripts/luma_color_map.py)
- **笔记本**: [`luma_color_map.ipynb`](luma_color_map.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SW26010/Texture-Map-Toolbox/blob/main/luma_color_map.ipynb)
- **文档**: [`docs/luma_color_map.md`](docs/luma_color_map.md)

### 2. 金属度/光滑度贴图生成 (Metallic Smoothness Map)

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
| Pillow | 图像读写 |
