"""Matplotlib-based Oklch state curve editor."""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

from texture_map_toolbox.core.luma import (
    DEFAULT_FAST_LUT_SIZE,
    DEFAULT_FAST_PREVIEW_SCALE,
    DITHER_STRENGTH,
    OklchCurveModel,
    STATE_CURVE_CTRL_POINTS,
    apply_luma_preview_lut,
    apply_precurve_dither,
    build_luma_preview_frame,
    build_luma_preview_lut,
    build_oklch_curve_model,
    build_state_curve_set,
    compute_luma_lut_indices,
    count_luma_preview_gamut_pixels,
    evaluate_reconstruction,
    load_image_data,
    load_state_curve_overrides,
    prepare_control_points,
    reconstruct_from_state_curves,
    resolve_dither_strength,
    resolve_input_image_path,
)
from texture_map_toolbox.gui.matplotlib_runtime import show_figures
from texture_map_toolbox.gui.luma_plots import plot_comparison


PREVIEW_SCALE = DEFAULT_FAST_PREVIEW_SCALE
PREVIEW_LUT_SIZE = DEFAULT_FAST_LUT_SIZE
CURVE_LINE_SAMPLES = 512
CURVE_X_DENSE = np.linspace(0.0, 1.0, CURVE_LINE_SAMPLES)


def _evaluate_hue_curve(
    hue_u_interp: PchipInterpolator,
    hue_v_interp: PchipInterpolator,
    x_values,
):
    """对 hue 状态曲线求值，返回 0-360°。"""
    hue_u = hue_u_interp(x_values)
    hue_v = hue_v_interp(x_values)
    norm = np.hypot(hue_u, hue_v)
    safe_norm = np.where(norm < 1e-8, 1.0, norm)
    return (np.degrees(np.arctan2(hue_v / safe_norm, hue_u / safe_norm)) + 360.0) % 360.0


class OklchCurveEditor:
    def __init__(
        self,
        image_path: str,
        rgb_float: np.ndarray,
        oklch_float: np.ndarray,
        valid_mask: np.ndarray,
        base_model: OklchCurveModel,
        *,
        initial_curve_overrides: dict | None = None,
        dither_strength: float | None = DITHER_STRENGTH,
        curve_output_path: str | None = None,
    ):
        self.image_path = image_path
        self.rgb_float = rgb_float
        self.oklch_float = oklch_float
        self.valid_mask = valid_mask
        self.base_model = base_model
        self.dither_strength = 0.0 if dither_strength is None else float(dither_strength)
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
        self.preview_frame = build_luma_preview_frame(
            self.rgb_float,
            self.oklch_float,
            self.valid_mask,
            preview_scale=PREVIEW_SCALE,
        )
        self.oklch_small = self.preview_frame.oklch_float
        self.mask_small = self.preview_frame.valid_mask
        self.y_small = self.preview_frame.y_image
        self.y_small_index = compute_luma_lut_indices(self.y_small, PREVIEW_LUT_SIZE)
        self._preview_output_mask = np.ones_like(self.mask_small, dtype=bool)
        self._output_valid_mask = np.ones_like(self.valid_mask, dtype=bool)
        self._preview_buf = np.empty((*self.y_small.shape, 3), dtype=np.uint8)

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

    def _build_preview_lut(self, state_curves, y_index):
        preview_lut_uint8, compressed_entries = build_luma_preview_lut(
            state_curves,
            preview_lut_size=PREVIEW_LUT_SIZE,
        )
        gamut_pixels = count_luma_preview_gamut_pixels(
            y_index,
            self.mask_small,
            compressed_entries,
        )
        return preview_lut_uint8, gamut_pixels

    def _sample_curve_lines(self, state_curves):
        lightness_line = np.clip(state_curves.lightness_interp(CURVE_X_DENSE), 0.0, 1.0)
        chroma_line = np.clip(state_curves.chroma_interp(CURVE_X_DENSE), 0.0, None)
        hue_line = _evaluate_hue_curve(state_curves.hue_u_interp, state_curves.hue_v_interp, CURVE_X_DENSE)
        return [lightness_line, chroma_line, hue_line]

    def _build_ui(self):
        self.fig = plt.figure(figsize=(16, 9))
        if self.fig.canvas.manager is not None:
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
        preview_y_eval = apply_precurve_dither(self.y_small, self.mask_small, self.dither_strength)
        if self.dither_strength > 0.0:
            preview_y_index = compute_luma_lut_indices(preview_y_eval, PREVIEW_LUT_SIZE)
        else:
            preview_y_index = self.y_small_index
        lut_uint8, gamut_pixels = self._build_preview_lut(state_curves, preview_y_index)
        mid = time.perf_counter()
        apply_luma_preview_lut(preview_y_index, self._preview_output_mask, lut_uint8, out_buf=self._preview_buf)
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
            output_valid_mask=self._output_valid_mask,
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
        show_figures(block=False)

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
        show_figures()


def launch_editor(
    image_path: str | None,
    *,
    alpha_mask_path: str | None = None,
    curve_path: str | None = None,
    curve_output_path: str | None = None,
    dither_strength: float | None = DITHER_STRENGTH,
) -> OklchCurveEditor:
    """构建并返回编辑器对象，供 CLI 或 GUI 复用。"""
    resolved_image_path = resolve_input_image_path(image_path)
    loaded_image = load_image_data(resolved_image_path, alpha_mask_path=alpha_mask_path)
    rgb_float = loaded_image.rgb_float
    oklch_float = loaded_image.oklch_float
    valid_mask = loaded_image.valid_mask
    resolved_dither_strength, _ = resolve_dither_strength(dither_strength, loaded_image)
    base_model, _ = build_oklch_curve_model(oklch_float, valid_mask)
    curve_overrides = load_state_curve_overrides(curve_path)
    editor = OklchCurveEditor(
        resolved_image_path,
        rgb_float,
        oklch_float,
        valid_mask,
        base_model,
        initial_curve_overrides=curve_overrides,
        dither_strength=resolved_dither_strength,
        curve_output_path=curve_output_path,
    )
    editor.image_warnings = loaded_image.image_warnings
    editor.alpha_source = loaded_image.alpha_source
    editor.alpha_mask_path = loaded_image.alpha_mask_path
    return editor


__all__ = [
    "CURVE_LINE_SAMPLES",
    "CURVE_X_DENSE",
    "OklchCurveEditor",
    "PREVIEW_LUT_SIZE",
    "PREVIEW_SCALE",
    "launch_editor",
]
