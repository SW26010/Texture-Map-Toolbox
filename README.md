# Texture-Map-Toolbox

围绕 Oklch luma 工作流构建的一组 Python 工具。当前仓库已经形成一条统一主线：

- `original`：离线高质量重建与评估
- `fast`：与 Qt 编辑器共享的快速 LUT 预览路径
- Qt 状态曲线编辑器：启动页、原图/预览图、三条曲线面板、目标图导入、mask 交互

## 当前能力

### Luma 工作流

- 输入轴统一使用原始 Oklch Lightness `L0`
- 基础模型拟合 `C(y)` 和 `h(y)`，其中 hue 通过单位圆向量插值规避 `0/360` 断裂
- 用户状态曲线显式建模为：
  - `L' = Lt(y)`
  - `C' = Ct(L')`
  - `h' = ht(L')`
- 保留 `original` 和 `fast` 两种算法入口
- 支持外部曲线 JSON、结果 JSON、输出图像保存

### Qt 编辑器

- 无参数启动时可直接打开 Qt 启动页选择原图、可选 alpha mask、初始 curves JSON 和导出路径
- 原图与预览图并排显示，预览与 `fast` 算法共享同一套 LUT helper
- 三条曲线都支持稀疏关键点编辑
- 内部关键点支持横向和纵向拖动，端点 `x` 锁定到 `0/1`
- 每条曲线都有 Key Points 数量调节和 Reset to Default
- `Ct(L')` 和 `ht(L')` 默认显示为稀疏 handles，但默认模式下实际仍回退到基础模型的精确基线
- 内置目标图导入对话框，可用一张图分别导入 `L / C / H`
- Hue 面板支持显示窗口起点滑块，可把显示范围平移到任意 `[start, start + 360]`

### Mask 语义

- 最高优先级：外部 `--alpha-mask` 或 GUI 里显式选择的 mask 文件
- 其次：图像本身带有“真正可用”的嵌入式 alpha
- Qt 交互路径下：若没有可用 mask，可改为用户点击一个或多个 seed 像素，按连通区域生成 mask
- 如果用户不提供额外 mask，也可以继续按整图有效的 implicit opaque 路径运行
- 旧的 border-connected auto-detect 代码仍保留在 core 中做参考、测试和非交互实验，但不再是 Qt 主交互路径

## 当前分层

- Core：`texture_map_toolbox.core.luma`
- API：`texture_map_toolbox.api.luma`
- GUI：`texture_map_toolbox.gui.editor`、`texture_map_toolbox.gui.qt_editor`、`texture_map_toolbox.gui.luma_plots`
- CLI：`texture_map_toolbox.__main__`、`texture_map_toolbox.cli.main`、`texture_map_toolbox.cli.luma`、`texture_map_toolbox.cli.editor`

详细说明见 [docs/architecture.md](docs/architecture.md) 和 [docs/cli.md](docs/cli.md)。

## 项目结构

```text
Texture-Map-Toolbox/
├── README.md
├── pyproject.toml
├── requirements.txt
├── luma_color_map.ipynb
├── metallic_smoothness_map.ipynb
├── docs/
│   ├── architecture.md
│   ├── cli.md
│   ├── hsl_curve_editor_design.md
│   ├── luma_color_map.md
│   ├── metallic_smoothness_map.md
│   ├── oklch_lut_design.md
│   └── examples/
├── tests/
│   ├── test_alpha_input_validation.py
│   ├── test_lightness_transfer_curve.py
│   ├── test_luma_smoke.py
│   └── test_qt_editor_smoke.py
└── texture_map_toolbox/
    ├── __main__.py
    ├── api/
    │   └── luma.py
    ├── cli/
    │   ├── editor.py
    │   ├── luma.py
    │   └── main.py
    ├── core/
    │   └── luma.py
    └── gui/
        ├── editor.py
        ├── luma_plots.py
        ├── matplotlib_runtime.py
        └── qt_editor.py
```

## 快速开始

```bash
pip install -r requirements.txt
```

如果你希望直接使用 console script：

```bash
pip install -e .
```

### 统一 CLI

```bash
# 查看命令结构
python -m texture_map_toolbox --help

# 无参数：直接打开 Qt 启动页
python -m texture_map_toolbox

# 原始高质量工作流
python -m texture_map_toolbox luma path/to/image.png --algorithm original

# 快速预览 LUT 工作流
python -m texture_map_toolbox luma path/to/image.png --algorithm fast --preview-scale 0.25 --preview-lut-size 512

# 使用外部 alpha mask
python -m texture_map_toolbox luma path/to/image.jpg --alpha-mask path/to/mask.png --algorithm original

# 使用曲线 JSON
python -m texture_map_toolbox luma path/to/image.png --curves path/to/curves.json --algorithm original

# 从 request JSON 执行
python -m texture_map_toolbox luma --request-json docs/examples/luma_request.fast.json

# Qt 编辑器
python -m texture_map_toolbox editor path/to/image.png --backend qt

# 无 image_path 时直接打开 Qt 启动页
python -m texture_map_toolbox editor --backend qt
```

安装为可编辑包后，也可以使用：

```bash
texture-map-toolbox luma path/to/image.png --algorithm original
texture-map-toolbox editor path/to/image.png --backend qt
```

## 编辑器概览

Qt 编辑器当前是推荐桌面入口，主要交互包括：

- 原图、预览图并排显示
- `Lt(y)`、`Ct(L')`、`ht(L')` 三个独立曲线面板
- 稀疏 Key Points 数量调节
- 控制点横向/纵向拖动
- Reset to Default 回到基础模型基线
- Export Image 导出当前全分辨率结果图像
- Hue 显示窗口起点滑块
- 一张目标图按 `L / C / H` 复选框分别导入
- 输入图和目标图的 mask 选择、connected-region seed preview、多种子 marker、颜色误差和区域偏移调节

更详细的交互说明见 [docs/hsl_curve_editor_design.md](docs/hsl_curve_editor_design.md)。

## 文档索引

- [docs/architecture.md](docs/architecture.md)：当前分层、依赖方向和推荐调用面
- [docs/cli.md](docs/cli.md)：CLI 结构、命令、JSON 协议和 GUI 入口
- [docs/luma_color_map.md](docs/luma_color_map.md)：luma 工作流原理与实现说明
- [docs/hsl_curve_editor_design.md](docs/hsl_curve_editor_design.md)：Qt/编辑器当前交互和设计
- [docs/oklch_lut_design.md](docs/oklch_lut_design.md)：Oklch LUT 方案设计与当前落地状态
- [docs/metallic_smoothness_map.md](docs/metallic_smoothness_map.md)：历史 metallic/smoothness 工作流说明

## 测试

当前 smoke 和行为测试覆盖 core、API、CLI、Qt 启动页、Qt 编辑器和 alpha 输入校验：

```bash
python -m unittest tests.test_luma_smoke tests.test_qt_editor_smoke tests.test_alpha_input_validation tests.test_lightness_transfer_curve
```

## API 与 GUI 集成

建议 GUI 或自动化流程直接复用 Python 层，而不是拼 shell 命令。

### API 调用

```python
from texture_map_toolbox.api.luma import LumaExecutionRequest, run_luma_workflow

result = run_luma_workflow(
    LumaExecutionRequest(
        image_path="path/to/image.png",
        algorithm="fast",
        show_plots=False,
    )
)
```

### Qt 启动页

```python
from texture_map_toolbox.gui.qt_editor import launch_qt_editor_launcher

launcher = launch_qt_editor_launcher(run_event_loop=False)
launcher.show()
```

### 直接打开 Qt 编辑器

```python
from texture_map_toolbox.gui.qt_editor import launch_qt_editor

launch_qt_editor("path/to/image.png")
```

## 备注

- 仓库内置样例图放在 `data/` 目录下；如果命令行没有显式传入图片路径，会优先尝试约定候选文件名，再回退到 `data/` 中第一个可用图片。
- `docs/examples/*.json` 中的 `image_path` 使用占位符，执行前请替换成实际路径。
- `metallic_smoothness_map.ipynb` 和对应文档目前是历史笔记本/说明页，不属于当前统一 CLI 主工作流。
