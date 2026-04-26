"""
交互式 Oklch 状态曲线编辑器

用法：
    python scripts/hsl_curve_editor.py [图片路径] [--curves 曲线文件]

操作：
    - 左键拖拽：移动控制点，实时更新预览
    - 右键点击控制点：重置该控制点到初始值
    - S / Ctrl+S：导出当前 Lt/Ct/ht 控制点到 JSON
    - Enter：执行一次全分辨率重建并显示结果
"""

import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

from luma_color_map import (
    DITHER_STRENGTH,
    STATE_CURVE_CTRL_POINTS,
    build_oklch_curve_model,
    build_state_curve_set,
    compress_oklch_chroma_to_srgb,
    evaluate_reconstruction,
    load_image,
    load_state_curve_overrides,
    plot_comparison,
    prepare_control_points,
    reconstruct_from_state_curves,
)


PREVIEW_SCALE = 0.25
PREVIEW_LUT_SIZE = 512
CURVE_LINE_SAMPLES = 512
CURVE_X_DENSE = np.linspace(0.0, 1.0, CURVE_LINE_SAMPLES)


def _evaluate_hue_curve(hue_u_interp: PchipInterpolator, hue_v_interp: PchipInterpolator, x_values):
    """对 hue 状态曲线求值，返回 0-360°。"""
    hue_u = hue_u_interp(x_values)
    hue_v = hue_v_interp(x_values)
    norm = np.hypot(hue_u, hue_v)
    safe_norm = np.where(norm < 1e-8, 1.0, norm)
    return (np.degrees(np.arctan2(hue_v / safe_norm, hue_u / safe_norm)) + 360.0) % 360.0


def recolor_fast(y_index_small, lut_uint8, invalid_mask_small, out_buf):
    """预览专用：复用缓冲区做 LUT 查表。"""
    np.take(lut_uint8, y_index_small, axis=0, out=out_buf)
    out_buf[invalid_mask_small] = 0


class OklchCurveEditor:
    def __init__(
        self,
        image_path: str,
        rgb_float: np.ndarray,
        oklch_float: np.ndarray,
        valid_mask: np.ndarray,
        base_model,
        *,
        initial_curve_overrides: dict | None = None,
        dither_strength: float = DITHER_STRENGTH,
        curve_output_path: str | None = None,
    ):
        self.image_path = image_path
        self.rgb_float = rgb_float
        self.oklch_float = oklch_float
        self.valid_mask = valid_mask
        self.base_model = base_model
        self.dither_strength = dither_strength
        self.curve_output_path = curve_output_path or self._default_curve_output_path()

        self._build_preview_inputs()
        self._initialize_controls(initial_curve_overrides or {})
        self._drag_state = None
        self._last_state_curves = None

        self._build_ui()
        self._render_preview()
        self._connect_events()

    def _default_curve_output_path(self) -> str:
        stem, _ = os.path.splitext(self.image_path)
        return f"{stem}_state_curves.json"

    def _build_preview_inputs(self):
        height, width = self.oklch_float.shape[:2]
        preview_height = max(1, int(height * PREVIEW_SCALE))
        preview_width = max(1, int(width * PREVIEW_SCALE))
        step_y = max(1, height // preview_height)
        step_x = max(1, width // preview_width)

        self.oklch_small = self.oklch_float[::step_y, ::step_x]
        self.mask_small = self.valid_mask[::step_y, ::step_x]
        self.y_small = self.oklch_small[:, :, 0]
        self.y_small_index = np.clip(
            np.round(self.y_small * (PREVIEW_LUT_SIZE - 1)), 0, PREVIEW_LUT_SIZE - 1
        ).astype(np.int32)
        self._preview_buf = np.empty((*self.y_small.shape, 3), dtype=np.uint8)
        self._invalid_mask_small = ~self.mask_small

    def _sample_initial_curves(self, initial_curve_overrides: dict):
        default_lightness_x = np.linspace(0.0, 1.0, STATE_CURVE_CTRL_POINTS)
        default_lightness_points = prepare_control_points(
            initial_curve_overrides.get("lightness_control_points"),
            default_lightness_x,
            default_lightness_x,
            clip_min=0.0,
            clip_max=1.0,
        )
        default_chroma_points = prepare_control_points(
            initial_curve_overrides.get("chroma_control_points"),
            self.base_model.key_y,
            self.base_model.key_c,
            clip_min=0.0,
        )
        default_hue_points = prepare_control_points(
            initial_curve_overrides.get("hue_control_points"),
            self.base_model.key_y,
            self.base_model.key_h,
            wrap_degrees=True,
        )
        return default_lightness_points, default_chroma_points, default_hue_points

    def _sample_base_model(self, x_values):
        chroma = np.clip(self.base_model.c_interp(x_values), 0.0, None)
        hue = _evaluate_hue_curve(self.base_model.u_interp, self.base_model.v_interp, x_values)
        return chroma, hue

    def _initialize_controls(self, initial_curve_overrides: dict):
        lightness_points, chroma_points, hue_points = self._sample_initial_curves(initial_curve_overrides)
        self.ctrl_x = [lightness_points[:, 0], chroma_points[:, 0], hue_points[:, 0]]
        self.ctrl_y = [lightness_points[:, 1], chroma_points[:, 1], hue_points[:, 1]]
        self.default_ctrl_y = [np.copy(values) for values in self.ctrl_y]

        base_chroma_max = max(np.max(self.base_model.key_c), np.max(chroma_points[:, 1]), 1e-3)
        self.chroma_ylim = (0.0, max(0.35, float(base_chroma_max) * 1.25))

        self.curve_cfg = [
            {
                "name": "Lightness Transfer Lt(y)",
                "color": "gold",
                "ylim": (0.0, 1.0),
                "ylabel": "Output Lightness L'",
                "xlabel": "Input Lightness L0",
            },
            {
                "name": "Chroma State Ct(L')",
                "color": "deepskyblue",
                "ylim": self.chroma_ylim,
                "ylabel": "Output Chroma C'",
                "xlabel": "Output Lightness L'",
            },
            {
                "name": "Hue State ht(L')",
                "color": "tomato",
                "ylim": (0.0, 360.0),
                "ylabel": "Output Hue h' (deg)",
                "xlabel": "Output Lightness L'",
            },
        ]

    def _current_control_point_payload(self) -> dict:
        return {
            "lightness_control_points": np.column_stack([self.ctrl_x[0], self.ctrl_y[0]]),
            "chroma_control_points": np.column_stack([self.ctrl_x[1], self.ctrl_y[1]]),
            "hue_control_points": np.column_stack([self.ctrl_x[2], self.ctrl_y[2]]),
        }

    def _build_state_curves(self):
        return build_state_curve_set(self.base_model, **self._current_control_point_payload())

    def _build_preview_lut(self, state_curves):
        preview_x = np.linspace(0.0, 1.0, PREVIEW_LUT_SIZE)
        lightness = np.clip(state_curves.lightness_interp(preview_x), 0.0, 1.0)
        chroma = np.clip(state_curves.chroma_interp(lightness), 0.0, None)
        hue = _evaluate_hue_curve(state_curves.hue_u_interp, state_curves.hue_v_interp, lightness)

        preview_oklch = np.column_stack([lightness, chroma, hue])
        _, preview_rgb, gamut_pixels = compress_oklch_chroma_to_srgb(preview_oklch)
        preview_lut_uint8 = np.clip(np.round(preview_rgb * 255.0), 0, 255).astype(np.uint8)
        return preview_lut_uint8, gamut_pixels

    def _sample_curve_lines(self, state_curves):
        lightness_line = np.clip(state_curves.lightness_interp(CURVE_X_DENSE), 0.0, 1.0)
        chroma_line = np.clip(state_curves.chroma_interp(CURVE_X_DENSE), 0.0, None)
        hue_line = _evaluate_hue_curve(state_curves.hue_u_interp, state_curves.hue_v_interp, CURVE_X_DENSE)
        return [lightness_line, chroma_line, hue_line]

    def _build_ui(self):
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title("Oklch State Curve Editor")

        grid = self.fig.add_gridspec(
            3,
            2,
            width_ratios=[1, 1.2],
            hspace=0.4,
            wspace=0.35,
            left=0.07,
            right=0.97,
            top=0.95,
            bottom=0.08,
        )
        self.curve_axes = [self.fig.add_subplot(grid[i, 0]) for i in range(3)]
        self.preview_ax = self.fig.add_subplot(grid[:, 1])

        self.fig.text(
            0.07,
            0.02,
            "Left drag: move point   Right click: reset point   S: save curves JSON   Enter: full-resolution render",
            fontsize=9,
        )

        self.curve_lines = []
        self.ctrl_scatters = []
        for index, (axis, cfg, ctrl_values) in enumerate(zip(self.curve_axes, self.curve_cfg, self.ctrl_y)):
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(*cfg["ylim"])
            axis.set_title(cfg["name"], fontsize=10)
            axis.set_ylabel(cfg["ylabel"], fontsize=8)
            axis.set_xlabel(cfg["xlabel"], fontsize=8)
            axis.grid(True, alpha=0.3)

            line, = axis.plot(CURVE_X_DENSE, np.zeros_like(CURVE_X_DENSE), color=cfg["color"], lw=2)
            scatter = axis.scatter(
                self.ctrl_x[index],
                ctrl_values,
                s=55 if ctrl_values.size <= STATE_CURVE_CTRL_POINTS else 14,
                color="white",
                edgecolors=cfg["color"],
                zorder=5,
                picker=True,
            )
            if index == 0:
                axis.plot([0, 1], [0, 1], color="gray", lw=0.8, linestyle="--", alpha=0.5)

            self.curve_lines.append(line)
            self.ctrl_scatters.append(scatter)

        self.preview_ax.set_title("Preview", fontsize=11)
        self.preview_ax.axis("off")
        self.preview_img = self.preview_ax.imshow(np.zeros((*self.y_small.shape, 3), dtype=np.uint8))

    def _render_preview(self):
        start = time.perf_counter()
        state_curves = self._build_state_curves()
        line_values = self._sample_curve_lines(state_curves)
        lut_uint8, gamut_pixels = self._build_preview_lut(state_curves)
        mid = time.perf_counter()
        recolor_fast(self.y_small_index, lut_uint8, self._invalid_mask_small, self._preview_buf)
        recolor_done = time.perf_counter()
        self.preview_img.set_data(self._preview_buf)

        for line, values in zip(self.curve_lines, line_values):
            line.set_ydata(values)
        self.fig.canvas.draw_idle()
        draw_done = time.perf_counter()

        self._last_state_curves = state_curves
        print(
            f"state+lut={1000 * (mid - start):.1f}ms  "
            f"recolor={1000 * (recolor_done - mid):.1f}ms  "
            f"draw={1000 * (draw_done - recolor_done):.1f}ms  "
            f"gamut={gamut_pixels}"
        )

    def _update_control_scatter(self, curve_idx):
        offsets = np.column_stack([self.ctrl_x[curve_idx], self.ctrl_y[curve_idx]])
        self.ctrl_scatters[curve_idx].set_offsets(offsets)

    def _save_curves(self):
        payload = {
            "lightness": np.column_stack([self.ctrl_x[0], self.ctrl_y[0]]).tolist(),
            "chroma": np.column_stack([self.ctrl_x[1], self.ctrl_y[1]]).tolist(),
            "hue": np.column_stack([self.ctrl_x[2], self.ctrl_y[2]]).tolist(),
        }
        with open(self.curve_output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"Saved curves: {self.curve_output_path}")

    def _render_full_resolution(self):
        state_curves = self._last_state_curves or self._build_state_curves()
        recolored_rgb_float, _, y_eval, gamut_pixels = reconstruct_from_state_curves(
            self.oklch_float,
            self.valid_mask,
            state_curves,
            dither_strength=self.dither_strength,
        )
        recolored_rgb_int, psnr, delta_e_image, delta_e_stats = evaluate_reconstruction(
            self.rgb_float,
            recolored_rgb_float,
            self.valid_mask,
        )

        print(f"Full-resolution gamut-compressed pixels: {gamut_pixels}")
        print(f"Full-resolution PSNR: {psnr:.2f} dB")
        for key, value in delta_e_stats.items():
            print(f"Delta E 2000 ({key.replace('_', ' ').title()}): {value:.2f}")

        plot_comparison(self.rgb_float, y_eval, recolored_rgb_int, self.valid_mask, psnr, delta_e_image)
        plt.show(block=False)

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    def _hit_test(self, event):
        for curve_idx, axis in enumerate(self.curve_axes):
            if event.inaxes is not axis or event.xdata is None or event.ydata is None:
                continue

            bbox = axis.get_window_extent()
            width = bbox.width
            height = bbox.height
            x_range = axis.get_xlim()
            y_range = axis.get_ylim()

            def to_px(x_value, y_value):
                px = (x_value - x_range[0]) / (x_range[1] - x_range[0]) * width
                py = (y_value - y_range[0]) / (y_range[1] - y_range[0]) * height
                return px, py

            event_x, event_y = to_px(event.xdata, event.ydata)
            for point_idx, (ctrl_x, ctrl_y) in enumerate(zip(self.ctrl_x[curve_idx], self.ctrl_y[curve_idx])):
                point_x, point_y = to_px(ctrl_x, ctrl_y)
                if (event_x - point_x) ** 2 + (event_y - point_y) ** 2 < 12 ** 2:
                    return curve_idx, point_idx
        return None

    def _on_press(self, event):
        hit = self._hit_test(event)
        if hit is None:
            return

        curve_idx, point_idx = hit
        if event.button == 3:
            self.ctrl_y[curve_idx][point_idx] = self.default_ctrl_y[curve_idx][point_idx]
            self._update_control_scatter(curve_idx)
            self._render_preview()
        elif event.button == 1:
            self._drag_state = (curve_idx, point_idx)

    def _on_release(self, _event):
        self._drag_state = None

    def _on_motion(self, event):
        if self._drag_state is None:
            return

        curve_idx, point_idx = self._drag_state
        axis = self.curve_axes[curve_idx]
        if event.inaxes is not axis or event.ydata is None:
            return

        y_min, y_max = self.curve_cfg[curve_idx]["ylim"]
        self.ctrl_y[curve_idx][point_idx] = float(np.clip(event.ydata, y_min, y_max))
        self._update_control_scatter(curve_idx)
        self._render_preview()

    def _on_key_press(self, event):
        if event.key in {"s", "ctrl+s"}:
            self._save_curves()
        elif event.key == "enter":
            self._render_full_resolution()

    def show(self):
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Oklch state-curve editor.")
    parser.add_argument(
        "image_path",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "mtmtPonyTail.png"),
        help="Input image path.",
    )
    parser.add_argument(
        "--curves",
        dest="curve_path",
        help="Optional JSON file containing initial Lt/Ct/ht control points.",
    )
    parser.add_argument(
        "--curve-output",
        dest="curve_output_path",
        help="Optional JSON file path used when exporting curves from the editor.",
    )
    parser.add_argument(
        "--dither-strength",
        type=float,
        default=DITHER_STRENGTH,
        help="Optional pre-curve dither amplitude applied on the input lightness axis.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Loading: {args.image_path}")

    rgb_float, oklch_float, valid_mask = load_image(args.image_path)
    base_model, _ = build_oklch_curve_model(oklch_float, valid_mask)
    curve_overrides = load_state_curve_overrides(args.curve_path)

    print("Building Oklch base model done. Opening editor...")
    editor = OklchCurveEditor(
        args.image_path,
        rgb_float,
        oklch_float,
        valid_mask,
        base_model,
        initial_curve_overrides=curve_overrides,
        dither_strength=args.dither_strength,
        curve_output_path=args.curve_output_path,
    )
    editor.show()