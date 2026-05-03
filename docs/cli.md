# CLI 结构整理

## 目标

CLI 当前只负责命令行交互：解析参数、打印摘要、读写 request/result JSON、调度 GUI 或算法执行。算法和桌面交互都已经拆分到独立层。

统一入口：

```bash
python -m texture_map_toolbox <command> [options]
```

如果不带任何参数：

```bash
python -m texture_map_toolbox
```

当前会直接打开 Qt 启动页，用图形界面选择原图、可选 alpha mask、初始 curves JSON 和导出路径，再进入 Qt 编辑器。

如果通过 `pip install -e .` 安装，也可以使用：

```bash
texture-map-toolbox <command> [options]
```

## 命令结构

| 命令 | 模块 | 用途 |
|------|------|------|
| `luma` | `texture_map_toolbox.cli.luma` | 运行 `original` / `fast` 两种 Oklch luma 工作流 |
| `editor` | `texture_map_toolbox.cli.editor` | 打开状态曲线编辑器，支持 `matplotlib` 与 Qt 两种后端 |

## 分层调用面

| 层 | 模块 | 责任 |
|----|------|------|
| Core | `texture_map_toolbox.core.luma` | 数值算法、mask 语义、预览 LUT、重建、结果对象 |
| API | `texture_map_toolbox.api.luma` | 稳定程序化调用面 |
| GUI | `texture_map_toolbox.gui.editor` / `texture_map_toolbox.gui.qt_editor` / `texture_map_toolbox.gui.luma_plots` | 编辑器交互与可视化 |
| CLI | `texture_map_toolbox.__main__` / `texture_map_toolbox.cli.main` / `texture_map_toolbox.cli.luma` / `texture_map_toolbox.cli.editor` | 参数解析、stdout 输出、命令调度 |

CLI 结构仍保持三层 helper：

- `configure_cli_parser(parser)`：把参数挂到已有 parser
- `parse_args(argv=None)`：解析该命令自己的参数
- `execute_cli(args)`：执行已解析命令

## `luma` 命令

### 示例

```bash
python -m texture_map_toolbox luma path/to/image.png --algorithm original
python -m texture_map_toolbox luma path/to/image.png --algorithm fast --preview-scale 0.25 --preview-lut-size 512
python -m texture_map_toolbox luma path/to/image.png --curves path/to/curves.json --algorithm original
python -m texture_map_toolbox luma path/to/image.jpg --alpha-mask path/to/mask.png --algorithm original
python -m texture_map_toolbox luma --request-json docs/examples/luma_request.fast.json
```

### 当前选项

- `--algorithm {original,fast}`：
	- `original` 保留离线高质量主流程
	- `fast` 使用与 Qt 编辑器共享的快速预览 LUT 路径
- `--alpha-mask path/to/mask.png`：用同尺寸二值/灰度 mask 覆盖图像自带 alpha
- `--curves path/to/curves.json`：读取 `Lt / Ct / ht` 控制点
- `--dither-strength`：输入轴预曲线抖动幅度
- `--preview-scale` / `--preview-lut-size`：控制 `fast` 模式
- `--output-image`：保存输出图像
- `--result-json` / `--summary-json`：输出结果摘要 JSON
- `--request-json`：从统一 request JSON 读入
- `--no-plots`：不弹出 matplotlib 图表
- `--skip-evaluation`：跳过 PSNR / Delta E 评估

### `luma` 的 mask 语义

命令行工作流不会交互式询问 seed mask。当前顺序是：

1. 外部 `--alpha-mask`
2. 可用的嵌入式 alpha
3. implicit opaque 继续运行

补充说明：

- PNG 全 1 alpha 会被视作“没有可用 mask”
- JPG、没有 alpha 的图片、以及全 1 PNG alpha 都会产生 warning
- 旧的 auto-detect helper 仍保留在 core 中，但当前 CLI 没有把它作为默认参数面暴露出来

## `editor` 命令

### 示例

```bash
python -m texture_map_toolbox editor
python -m texture_map_toolbox editor path/to/image.png --backend matplotlib
python -m texture_map_toolbox editor path/to/image.png --backend qt
python -m texture_map_toolbox editor path/to/image.png --alpha-mask path/to/mask.png --curves path/to/curves.json --curve-output path/to/output.json --backend qt
python -m texture_map_toolbox editor --backend qt
```

### 当前选项

- `image_path`：可选；当存在本地样例图时可以省略
- `--alpha-mask`：可选外部 mask
- `--curves`：可选初始曲线 JSON
- `--curve-output`：保存曲线 JSON 的默认路径
- `--dither-strength`：输入轴预曲线抖动幅度
- `--backend {matplotlib,qt}`：选择编辑器后端，默认是 `matplotlib`

### Qt 后端行为

- `python -m texture_map_toolbox`：直接打开 Qt 启动页
- `python -m texture_map_toolbox editor --backend qt` 且未提供 `image_path`：也会打开 Qt 启动页
- `python -m texture_map_toolbox editor path/to/image.png --backend qt`：直接加载图片并进入编辑器

### Qt 编辑器当前功能

- 原图与预览图并排显示
- `Lt(y)`、`Ct(L')`、`ht(L')` 三个曲线面板
- 稀疏关键点编辑，内部点支持横向/纵向拖动
- 每条曲线独立的 Key Points 数量调节
- 每条曲线独立的 Reset to Default
- `Lt(y)` 灰度背景与输出 Lightness 直方图叠加
- `Ct(L')` / `ht(L')` 动态色彩背景
- 内置目标图选择器，一张图可用复选框分别导入 `L / C / H`
- Hue 显示窗口起点滑块，可把显示范围设为任意 `[start, start + 360]`

### Qt 路径下的 mask 语义

Qt 编辑器和目标图对话框支持交互式 mask 路径：

1. 外部 mask 文件
2. 可用嵌入式 alpha
3. 用户点击一个或多个 seed 像素，按 connected-region 生成 mask
4. Continue Without Extra Mask，按整图有效继续

connected-region seed mask 当前支持：

- 多个 seed 点
- 颜色误差滑块
- 区域偏移滑块
- 原图 marker 预览
- mask preview 面板

## request / result JSON

### request JSON

`luma` 支持从 JSON 直接读取请求。当前支持字段：

- `image_path`
- `alpha_mask_path`
- `curve_path`
- `algorithm`
- `dither_strength`
- `evaluate_result`
- `show_plots`
- `preview_scale`
- `preview_lut_size`
- `output_image_path`
- `result_json_path`

示例：

- [docs/examples/luma_request.original.json](docs/examples/luma_request.original.json)
- [docs/examples/luma_request.fast.json](docs/examples/luma_request.fast.json)

### result JSON

无论执行 `original` 还是 `fast`，结果摘要都会包含统一字段，例如：

- `algorithm`
- `image_path`
- `alpha_mask_path`
- `alpha_source`
- `image_warnings`
- `curve_path`
- `curve_source`
- `dither_strength`
- `source_image_shape`
- `output_image_shape`
- `output_scale`
- `preview_lut_size`
- `keypoints`
- `state_curve_points`
- `gamut_compressed_pixels`
- `evaluation_enabled`
- `output_image_path`

其中：

- `fast` 会额外包含 `gamut_compressed_lut_entries`
- 开启评估时会追加 `psnr` 和 `delta_e_stats`

## GUI / 集成建议

如果后续继续做 GUI 或自动化流程，建议直接复用 Python 入口，不要从 GUI 中拼 shell 字符串：

1. 用 `texture_map_toolbox.api.luma.run_luma_workflow(...)` 跑统一工作流。
2. 用 `texture_map_toolbox.gui.qt_editor.launch_qt_editor_launcher(...)` 打开 Qt 启动页。
3. 用 `texture_map_toolbox.gui.qt_editor.launch_qt_editor(...)` 直接进入 Qt 编辑器。
4. 如需兼容旧版交互，再调用 `texture_map_toolbox.gui.editor.launch_editor(...)`。
5. 如需绘图，直接复用 `texture_map_toolbox.gui.luma_plots`。