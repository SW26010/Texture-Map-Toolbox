# CLI 结构整理

## 目标

CLI 现在只承担命令行交互职责：解析参数、打印摘要、触发 GUI 展示。算法执行和 GUI 实现已经拆到单独层，便于后续继续演进桌面 GUI 或其他调用方式。

推荐入口：

```bash
python -m texture_map_toolbox <command> [options]
```

安装为可编辑包后，也可以使用：

```bash
texture-map-toolbox <command> [options]
```

## 命令结构

| 命令 | 模块 | 用途 |
|------|------|------|
| `luma` | `texture_map_toolbox.cli.luma` | 统一运行 `original` / `fast` 两种 luma 算法，并支持 request/result JSON |
| `editor` | `texture_map_toolbox.cli.editor` | Oklch 状态曲线编辑器，支持 `matplotlib` 和 Qt MVP 两种后端 |

## 分层调用面

| 层 | 模块 | 责任 |
|----|------|------|
| Core | `texture_map_toolbox.core.luma` | 数值算法、数据对象、JSON / 图像 helper |
| API | `texture_map_toolbox.api.luma` | 稳定程序化调用面 |
| GUI | `texture_map_toolbox.gui.editor` / `texture_map_toolbox.gui.luma_plots` | 编辑器交互与 matplotlib 可视化 |
| CLI | `texture_map_toolbox.__main__` / `texture_map_toolbox.cli.main` / `texture_map_toolbox.cli.luma` / `texture_map_toolbox.cli.editor` | 参数解析、stdout 输出、命令调度 |

CLI 解析与执行仍然保持拆分：

- `configure_cli_parser(parser)`：把参数挂到已有 parser 上
- `parse_args(argv=None)`：直接解析该工具自己的参数
- `execute_cli(args)`：执行已解析命令
- `main(argv=None)`：完整 CLI 入口

## 命令示例

### 1. Oklch 主流程

```bash
python -m texture_map_toolbox luma path/to/image.png --algorithm original
python -m texture_map_toolbox luma path/to/image.png --algorithm original --curves path/to/curves.json
python -m texture_map_toolbox luma path/to/image.png --algorithm fast --preview-scale 0.25 --preview-lut-size 512
python -m texture_map_toolbox luma --request-json docs/examples/luma_request.fast.json
```

其中：

- `--algorithm original` 保留原始离线高质量算法
- `--algorithm fast` 使用与未来 GUI 共享的快速 LUT 算法
- `--no-plots` 禁止弹出 matplotlib 图表
- `--skip-evaluation` 跳过 PSNR / Delta E 评估
- `--output-image` 保存当前算法输出图像
- `--result-json` / `--summary-json` 把本次运行摘要写成 JSON，适合后续 GUI 或外部流程读取
- `--request-json` 从统一 request JSON 读取参数
- `--preview-scale` / `--preview-lut-size` 控制快速算法的缩放和 LUT 采样数

### 2. 状态曲线编辑器

```bash
python -m texture_map_toolbox editor
python -m texture_map_toolbox editor path/to/image.png --curves path/to/curves.json --curve-output path/to/output.json
python -m texture_map_toolbox editor path/to/image.png --backend qt
```

其中：

- `--backend matplotlib` 保留当前 matplotlib 编辑器
- `--backend qt` 打开新的 Qt MVP 编辑器

Qt MVP 当前包括：

- 原图和预览图并排显示
- `Lt(y)` / `Ct(L')` / `ht(L')` 三张曲线图
- 三组控制点的拖拽调整
- 与当前 `fast` 预览路径共享的实时预览
- 全分辨率确认按钮和曲线 JSON 导出按钮

## 统一 request / result JSON

### request JSON

`luma` 命令支持直接从 JSON 文件读取执行请求。当前支持字段：

- `image_path`
- `curve_path`
- `algorithm`: `original` 或 `fast`
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

示例 JSON 中的 `image_path` 使用占位路径，调用前请先替换成实际输入图路径。

### result JSON

result JSON 会返回统一摘要，不论执行的是 `original` 还是 `fast`，都会包含：

- `algorithm`
- `image_path`
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

- `fast` 模式会额外包含 `gamut_compressed_lut_entries`
- 开启评估时会追加 `psnr` 和 `delta_e_stats`

## GUI / 集成建议

如果后续要做 GUI，建议直接复用 Python 入口，不要从 GUI 里拼 shell 字符串：

1. 通过 `texture_map_toolbox.api.luma.run_luma_workflow(...)` 跑统一工作流。
2. 通过 `texture_map_toolbox.gui.editor.launch_editor(...)` 打开状态曲线编辑器。
3. 如果需要 Qt MVP，调用 `texture_map_toolbox.gui.qt_editor.launch_qt_editor(...)`。
4. 如果需要绘图展示，调用 `texture_map_toolbox.gui.luma_plots`。
5. 如果仍希望走命令行协议，统一调用 `texture_map_toolbox.__main__.main(argv)`，并优先使用 request/result JSON 交换数据。