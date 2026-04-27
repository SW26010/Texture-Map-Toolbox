# CLI 结构整理

## 目标

当前仓库已经把命令行入口整理成一层统一的 CLI，供后续 GUI 复用同一套调用语义、默认值和参数解析逻辑。

推荐入口：

```bash
python -m scripts.cli <command> [options]
```

保留兼容入口：

- `python scripts/luma_color_map.py ...`
- `python scripts/hsl_curve_editor.py ...`

这些脚本现在都是统一 CLI 语义下的包装层。

## 命令结构

| 命令 | 模块 | 用途 |
|------|------|------|
| `luma` | `scripts/luma_color_map.py` | 统一运行 `original` / `fast` 两种 luma 算法，并支持 request/result JSON |
| `editor` | `scripts/hsl_curve_editor.py` | Oklch 状态曲线编辑器 |

## 程序入口

为了 GUI 复用，三个工具都暴露了明确的程序入口函数：

| 模块 | GUI / 集成建议调用面 |
|------|----------------------|
| `scripts/luma_color_map.py` | `run_luma_color_map(...)` / `run_luma_workflow(...)` |
| `scripts/hsl_curve_editor.py` | `launch_editor(...)` |
| `scripts/cli.py` | `main(argv=None)` |

CLI 解析与执行也被拆开了：

- `configure_cli_parser(parser)`：把参数挂到已有 parser 上
- `parse_args(argv=None)`：直接解析该工具自己的参数
- `execute_cli(args)`：执行已解析命令
- `main(argv=None)`：完整 CLI 入口

这样的结构对 GUI 更友好，因为 GUI 不必模拟 shell，也不必复制参数默认值。

## 命令示例

### 1. Oklch 主流程

```bash
python -m scripts.cli luma path/to/image.png --algorithm original
python -m scripts.cli luma path/to/image.png --algorithm original --curves path/to/curves.json
python -m scripts.cli luma path/to/image.png --algorithm fast --preview-scale 0.25 --preview-lut-size 512
python -m scripts.cli luma --request-json docs/examples/luma_request.fast.json
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

### 2. 状态曲线编辑器

```bash
python -m scripts.cli editor
python -m scripts.cli editor path/to/image.png --curves path/to/curves.json --curve-output path/to/output.json
```

## GUI 准备建议

如果后续要做 GUI，建议直接复用 Python 入口，不要从 GUI 里拼 shell 字符串：

1. 通过 `run_luma_color_map(..., algorithm="original")` 跑原始高质量算法。
2. 通过 `run_luma_color_map(..., algorithm="fast")` 跑与 GUI 共享的快速 LUT 算法。
3. 通过 `launch_editor(...)` 打开状态曲线编辑器。
4. 如果 GUI 仍希望走命令行协议，统一调用 `scripts.cli.main(argv)`，并优先使用 request/result JSON 交换数据。