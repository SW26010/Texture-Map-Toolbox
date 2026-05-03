# Texture-Map-Toolbox

一组围绕 Oklch luma LUT 工作流构建的 Python 工具，当前重点是离线重建、可复用 core/API，以及后续可继续扩展的 GUI 交互层。

## 当前分层

- Core：`texture_map_toolbox.core.luma`
- API：`texture_map_toolbox.api.luma`
- GUI：`texture_map_toolbox.gui.editor`、`texture_map_toolbox.gui.luma_plots`
- CLI：`texture_map_toolbox.__main__`、`texture_map_toolbox.cli.luma`、`texture_map_toolbox.cli.editor`

详细分层说明见 [`docs/architecture.md`](docs/architecture.md) 和 [`docs/cli.md`](docs/cli.md)。

## 项目结构

```
Texture-Map-Toolbox/
├── README.md                              # 项目说明
├── requirements.txt                       # Python 依赖
├── pyproject.toml                         # 包配置与 console script
├── tests/
│   └── test_luma_smoke.py                 # 基于样例图的 smoke tests
├── texture_map_toolbox/
│   ├── __main__.py                        # 标准 package 入口
│   ├── api/
│   │   └── luma.py                        # 稳定 API 调用面
│   ├── cli/
│   │   ├── main.py                        # 统一 CLI 入口
│   │   ├── luma.py                        # luma CLI 适配层
│   │   └── editor.py                      # editor CLI 适配层
│   ├── core/
│   │   └── luma.py                        # 数值核心实现
│   └── gui/
│       ├── editor.py                      # matplotlib 编辑器 GUI
│       ├── luma_plots.py                  # matplotlib 绘图层
│       └── matplotlib_runtime.py          # matplotlib 运行时辅助
├── docs/
│   ├── architecture.md                    # 分层结构说明
│   ├── cli.md                             # CLI 与调用约定
│   ├── examples/                          # CLI request 示例
│   ├── hsl_curve_editor_design.md         # Oklch 状态曲线编辑器设计说明
│   ├── luma_color_map.md                  # 亮度色彩映射详细文档
│   └── oklch_lut_design.md                # Oklch LUT 设计草案
├── luma_color_map.ipynb                   # 亮度色彩映射交互式笔记本
└── metallic_smoothness_map.ipynb          # 历史笔记本，当前不作为主工作流
```

## 工具说明

### 1. 亮度色彩映射 (Luma Color Map)

从彩色图像出发，先转换到 Oklch，直接使用原始 Oklch 的 Lightness ($L_0$) 作为输入轴。当前统一接口下同时保留：

- `original`：原始离线高质量主流程
- `fast`：与未来 GUI 共享的快速 LUT 预览算法

- Core：[`texture_map_toolbox/core/luma.py`](texture_map_toolbox/core/luma.py)
- API：[`texture_map_toolbox/api/luma.py`](texture_map_toolbox/api/luma.py)
- 笔记本：[`luma_color_map.ipynb`](luma_color_map.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SW26010/Texture-Map-Toolbox/blob/main/luma_color_map.ipynb)
- 文档：[`docs/luma_color_map.md`](docs/luma_color_map.md)
- 设计草案：[`docs/oklch_lut_design.md`](docs/oklch_lut_design.md)

### 2. Oklch 状态曲线编辑器 (Oklch State Curve Editor)

交互式工具，用于编辑 `Lt(y)`、`Ct(L')` 和 `ht(L')` 三条状态曲线。编辑器会在缩小预览图上实时重着色，支持导出 JSON 控制点文件，并可手动触发一次全分辨率重建。

- GUI：[`texture_map_toolbox/gui/editor.py`](texture_map_toolbox/gui/editor.py)
- 设计说明：[`docs/hsl_curve_editor_design.md`](docs/hsl_curve_editor_design.md)

## 快速开始

```bash
pip install -r requirements.txt
```

如果你希望直接使用 console script，可以额外执行：

```bash
pip install -e .
```

推荐使用新的统一 CLI：

```bash
# 查看统一命令结构
python -m texture_map_toolbox --help

# 原始高质量算法
python -m texture_map_toolbox luma path/to/image.png --algorithm original

# 使用外部 Lt/Ct/ht 控制点
python -m texture_map_toolbox luma path/to/image.png --algorithm original --curves path/to/curves.json

# 与未来 GUI 共享的快速 LUT 算法
python -m texture_map_toolbox luma path/to/image.png --algorithm fast --preview-scale 0.25 --preview-lut-size 512

# 使用统一 request / result JSON
python -m texture_map_toolbox luma --request-json docs/examples/luma_request.fast.json

# Oklch 状态曲线编辑器
python -m texture_map_toolbox editor path/to/image.png --curves path/to/curves.json
```

安装为可编辑包后，也可以使用：

```bash
texture-map-toolbox luma path/to/image.png --algorithm original
texture-map-toolbox editor path/to/image.png --curves path/to/curves.json
```

仓库内置样例图放在 `data/` 目录下；如果命令行没有显式传入图片路径，工具会优先尝试约定的样例文件名，并在找不到时回退到 `data/` 目录里第一个可用图片。

`docs/examples/*.json` 中的 `image_path` 是占位符，使用前请替换成你的实际输入图路径。

## 测试

当前仓库包含一套基于样例图的 smoke tests，覆盖 core、API、CLI、绘图和编辑器初始化：

```bash
python -m unittest tests.test_luma_smoke
```

## API 与 GUI 集成

建议 GUI 或其他自动化流程直接复用 Python 层，而不是拼 shell 字符串。

### API 调用

```python
from texture_map_toolbox.api.luma import LumaExecutionRequest, run_luma_workflow

request = LumaExecutionRequest(
    image_path="path/to/image.png",
    algorithm="fast",
    show_plots=False,
)
result = run_luma_workflow(request)
```

### GUI 调用

```python
from texture_map_toolbox.gui.editor import launch_editor

editor = launch_editor("path/to/image.png", curve_path="path/to/curves.json")
editor.show()
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
