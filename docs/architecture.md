# 架构分层

当前仓库已经稳定成一套按职责分离的 Oklch 工具链：数值核心只负责算法与数据语义，API 负责稳定调用面，GUI 负责桌面交互和可视化，CLI 负责参数解析与调度。

## 层级划分

### 1. Core

- `texture_map_toolbox/core/luma.py`

这一层集中所有数值实现与共享数据结构，包括：

- 图像加载、alpha 解析、mask 来源判定
- Oklch 转换与样本云构建
- `C(y)` / `h(y)` 基础模型拟合
- `Lt / Ct / ht` 状态曲线构建与求值
- 原始高质量重建
- 共享的快速预览 LUT helper
- gamut compression、评估与结果摘要 helper

当前 Core 中的 mask 语义是：

- 外部 alpha mask 优先于图像自带 alpha
- PNG 全 1 alpha 会被视作“没有可用 mask”
- 支持用户选中的 connected-region seed mask
- 没有额外 mask 时可以按 implicit opaque 继续
- 旧的 border-connected auto-detect 仍保留为内部/测试参考，不再是 Qt 主交互路径

这一层不承担 argparse、Qt 或 matplotlib 的职责。

### 2. API

- `texture_map_toolbox/api/luma.py`

API 层提供稳定的 Python 集成入口，面向：

- GUI
- notebook
- 自动化脚本
- 后续其他服务化调用方

推荐优先使用：

- `LumaExecutionRequest`
- `LumaExecutionResult`
- `run_luma_workflow(...)`
- `fit_monotonic_lightness_transfer_curve(...)`
- `summarize_luma_result(...)`

API 本身不重复实现算法，只暴露稳定调用面和少量高层 helper。

### 3. GUI

- `texture_map_toolbox/gui/editor.py`
- `texture_map_toolbox/gui/qt_editor.py`
- `texture_map_toolbox/gui/luma_plots.py`
- `texture_map_toolbox/gui/matplotlib_runtime.py`

GUI 层承载两类职责：

1. 交互式桌面编辑
2. 分析/比较图绘制

当前有两个编辑器后端：

- `matplotlib`：历史版本，功能更少，主要保留兼容性
- Qt：当前推荐入口，包含启动页、connected-region seed mask、目标图导入、稀疏关键点、Hue 显示窗口等交互

Qt 编辑器当前直接复用 Core 的预览 LUT helper，不再单独维护一套独立预览算法。

### 4. CLI

- `texture_map_toolbox/__main__.py`
- `texture_map_toolbox/cli/main.py`
- `texture_map_toolbox/cli/luma.py`
- `texture_map_toolbox/cli/editor.py`

CLI 层只负责：

- 参数解析
- stdout 摘要输出
- request/result JSON 读写
- 调用 API 或 GUI

统一入口：

```bash
python -m texture_map_toolbox <command> [options]
```

如果不带参数：

```bash
python -m texture_map_toolbox
```

当前会直接打开 Qt 启动页。

## 依赖方向

当前建议的依赖方向如下：

```text
API -> core
GUI -> core
CLI -> API
CLI -> GUI
package entrypoint -> CLI
```

这样做的结果是：

- Core 不依赖命令行或 GUI 框架
- Qt/CLI 共用同一套算法语义
- GUI 不需要模拟 shell
- request/result 和曲线 JSON 可以稳定复用
- 未来如果继续更换前端框架，不需要推倒核心算法

## 共享数据语义

### 1. 输入轴与基础模型

系统统一使用原始 Oklch Lightness 作为输入轴：

$$
y = L_0
$$

基础模型拟合：

- `C(y)`
- `h(y)`

hue 内部通过单位圆向量拟合和求值，避免 `0/360` seam 断裂。

### 2. 状态曲线

用户状态曲线统一定义为：

$$
L' = L_t(y), \quad C' = C_t(L'), \quad h' = h_t(L')
$$

Qt 编辑器里默认模式与编辑模式的区别是：

- 默认模式：`Lt(y)` 为 identity，`Ct(L')` / `ht(L')` 仍回退到基础模型精确基线
- 编辑模式：曲线 JSON payload 中显式保存 override 控制点

### 3. 曲线 JSON

曲线文件使用统一 JSON 语义：

- `lightness`
- `chroma`
- `hue`

Qt 编辑器当前只保存已进入 override 模式的曲线；Reset to Default 后，对应曲线键会重新省略。

## 推荐调用面

### 运行 luma 工作流

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

### 打开 Qt 启动页

```python
from texture_map_toolbox.gui.qt_editor import launch_qt_editor_launcher

launch_qt_editor_launcher()
```

### 直接打开 Qt 编辑器

```python
from texture_map_toolbox.gui.qt_editor import launch_qt_editor

launch_qt_editor("path/to/image.png")
```

### 走统一 CLI

```python
from texture_map_toolbox.__main__ import main

exit_code = main(["luma", "path/to/image.png", "--algorithm", "fast"])
```

## 测试覆盖面

当前核心回归主要集中在：

- `tests/test_luma_smoke.py`
- `tests/test_alpha_input_validation.py`
- `tests/test_lightness_transfer_curve.py`
- `tests/test_qt_editor_smoke.py`

它们覆盖了：

- 默认样例图解析
- 原始/快速工作流 smoke
- alpha 输入语义与 connected-region seed mask
- Qt 启动页和 Qt 编辑器关键交互
- Lightness 目标曲线拟合