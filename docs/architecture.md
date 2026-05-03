# 架构分层

当前仓库已经把外部交互面整理成明确的分层结构，目的是让 CLI、GUI 和未来的其他调用方复用同一套算法与数据模型，而不是彼此复制流程。

## 层级划分

### 1. 数值核心

- [`texture_map_toolbox/core/luma.py`](../texture_map_toolbox/core/luma.py)

当前 Oklch luma 数值主链路集中在这里，包括：

- 图像加载与 Oklch 转换
- `C(y)` / `h(y)` 基础模型拟合
- `Lt / Ct / ht` 状态曲线求值
- 原始高质量算法与快速 LUT 算法
- 结果摘要和 JSON 输出 helper

这层不承担 CLI、GUI 或 matplotlib 交互职责，只负责算法、数据对象和序列化语义。

### 2. API 层

- [`texture_map_toolbox/api/luma.py`](../texture_map_toolbox/api/luma.py)

这一层提供稳定、可复用的 Python 调用面，面向：

- GUI
- 自动化脚本
- Notebook
- 未来 Web/API 服务

建议优先通过下面这些对象和函数集成：

- `LumaExecutionRequest`
- `LumaExecutionResult`
- `run_luma_workflow(...)`
- `run_luma_color_map(...)`
- `summarize_luma_result(...)`

API 层本身不再承载实现，只是对 core 的稳定导出面。

### 3. GUI 层

- [`texture_map_toolbox/gui/editor.py`](../texture_map_toolbox/gui/editor.py)
- [`texture_map_toolbox/gui/qt_editor.py`](../texture_map_toolbox/gui/qt_editor.py)
- [`texture_map_toolbox/gui/luma_plots.py`](../texture_map_toolbox/gui/luma_plots.py)

GUI 层专门承载交互和可视化职责，包括：

- matplotlib 状态曲线编辑器
- Qt MVP 状态曲线编辑器
- luma 比较图与分析图
- 全分辨率确认时的图形展示

GUI 层依赖 core，不直接重新实现算法。

### 4. CLI 层

- [`texture_map_toolbox/__main__.py`](../texture_map_toolbox/__main__.py)
- [`texture_map_toolbox/cli/main.py`](../texture_map_toolbox/cli/main.py)
- [`texture_map_toolbox/cli/luma.py`](../texture_map_toolbox/cli/luma.py)
- [`texture_map_toolbox/cli/editor.py`](../texture_map_toolbox/cli/editor.py)

CLI 层专门负责：

- 参数解析
- 控制台输出
- 将用户输入折叠成 request 对象
- 调用 API / GUI 层

推荐入口：

```bash
python -m texture_map_toolbox <command> [options]
```

如果通过 `pip install -e .` 安装，也可以使用：

```bash
texture-map-toolbox <command> [options]
```

## 依赖方向

当前建议的依赖方向如下：

```text
API -> core
GUI -> core
CLI -> API
CLI -> GUI
package entrypoint -> CLI
```

这样做的好处是：

- GUI 不必模拟 shell
- CLI 不再承载编辑器实现
- core 不再耦合 argparse 或 matplotlib
- 未来迁移到 PyQt、Dear PyGui 或 Web 前端时，不必改算法调用协议
- request/result 数据对象可以稳定复用

## 推荐调用面

### 运行算法

```python
from texture_map_toolbox.api.luma import LumaExecutionRequest, run_luma_workflow

result = run_luma_workflow(
    LumaExecutionRequest(
        image_path="path/to/image.png",
        algorithm="original",
        show_plots=False,
    )
)
```

### 打开编辑器

```python
from texture_map_toolbox.gui.editor import launch_editor

editor = launch_editor("path/to/image.png")
editor.show()
```

如果要使用 Qt MVP：

```python
from texture_map_toolbox.gui.qt_editor import launch_qt_editor

launch_qt_editor("path/to/image.png")
```

### 走命令行协议

```python
from texture_map_toolbox.__main__ import main

exit_code = main(["luma", "path/to/image.png", "--algorithm", "fast"])
```