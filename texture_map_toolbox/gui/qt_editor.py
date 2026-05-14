"""Qt-based MVP editor for the Oklch state curves."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pyqtgraph as pg
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.interpolate import PchipInterpolator

from texture_map_toolbox.core.luma import (
    DEFAULT_FAST_LUT_SIZE,
    DEFAULT_FAST_PREVIEW_SCALE,
    DEFAULT_SEED_MASK_COLOR_TOLERANCE,
    DITHER_STRENGTH,
    LoadedImageData,
    OklchCurveModel,
    STATE_CURVE_CTRL_POINTS,
    apply_luma_preview_lut,
    apply_precurve_dither,
    build_luma_preview_frame,
    build_luma_preview_lut,
    build_oklch_curve_model,
    build_state_curve_set,
    compress_oklch_chroma_to_srgb,
    compute_luma_lut_indices,
    count_luma_preview_gamut_pixels,
    detect_seeded_valid_mask,
    evaluate_reconstruction,
    fit_monotonic_lightness_transfer_curve,
    load_image_data,
    load_state_curve_overrides,
    prepare_control_points,
    reconstruct_from_state_curves,
    resolve_input_image_path,
    save_luma_output_image,
)
from texture_map_toolbox.gui.luma_plots import plot_comparison
from texture_map_toolbox.gui.matplotlib_runtime import show_figures


pg.setConfigOptions(antialias=True)

_OPEN_QT_TOP_LEVEL_WINDOWS: list[QtWidgets.QWidget] = []

PREVIEW_SCALE = DEFAULT_FAST_PREVIEW_SCALE
PREVIEW_LUT_SIZE = DEFAULT_FAST_LUT_SIZE
CURVE_LINE_SAMPLES = 512
CURVE_BACKGROUND_WIDTH = 512
CURVE_BACKGROUND_HEIGHT = 256
LIGHTNESS_HISTOGRAM_BINS = 96
CURVE_X_DENSE = np.linspace(0.0, 1.0, CURVE_LINE_SAMPLES)
PIL_PREVIEW_RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
CURVE_EDITOR_MIN_CTRL_POINTS = 2
CURVE_EDITOR_MAX_CTRL_POINTS = 512
CURVE_EDITOR_OVERRIDE_KEYS = (
    "lightness_control_points",
    "chroma_control_points",
    "hue_control_points",
)
CURVE_EDITOR_SAVE_KEYS = ("lightness", "chroma", "hue")
CURVE_EDITOR_MODE_DEFAULT = "Original curve active"
CURVE_EDITOR_MODE_EDITED = "Editable key points active"
HUE_DISPLAY_WINDOW_SPAN = 360.0
HUE_DISPLAY_START_MIN = 0
HUE_DISPLAY_START_MAX = 359
MASK_COLOR_TOLERANCE_MAX = 64
MASK_REGION_OFFSET_MIN = -64
MASK_REGION_OFFSET_MAX = 64
SEED_MARKER_FILL_COLOR = QtGui.QColor("#ff6b4a")
SEED_MARKER_OUTLINE_COLOR = QtGui.QColor("#ffffff")


def _evaluate_hue_curve(
    hue_u_interp: PchipInterpolator,
    hue_v_interp: PchipInterpolator,
    x_values,
):
    """Evaluate a hue state curve and return degrees in [0, 360)."""
    hue_u = hue_u_interp(x_values)
    hue_v = hue_v_interp(x_values)
    norm = np.hypot(hue_u, hue_v)
    safe_norm = np.where(norm < 1e-8, 1.0, norm)
    return (np.degrees(np.arctan2(hue_v / safe_norm, hue_u / safe_norm)) + 360.0) % 360.0


def _rgb_float_to_uint8(rgb_float: np.ndarray) -> np.ndarray:
    """Convert a float RGB image into uint8 RGB."""
    return np.clip(np.round(np.asarray(rgb_float) * 255.0), 0.0, 255.0).astype(np.uint8)


def _rgb_uint8_to_qimage(rgb_uint8: np.ndarray) -> QtGui.QImage:
    """Convert an RGB uint8 array into a detached QImage."""
    rgb_uint8 = np.ascontiguousarray(rgb_uint8)
    height, width, _ = rgb_uint8.shape
    image = QtGui.QImage(
        rgb_uint8.data,
        width,
        height,
        width * 3,
        QtGui.QImage.Format.Format_RGB888,
    )
    return image.copy()


def _build_polyline_adjacency(point_count: int) -> np.ndarray:
    """Build segment connectivity for a polyline graph item."""
    if point_count < 2:
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack([
        np.arange(point_count - 1, dtype=np.int32),
        np.arange(1, point_count, dtype=np.int32),
    ])


def _ensure_qt_application() -> tuple[QtWidgets.QApplication, bool]:
    """Return the active QApplication and whether it was created here."""
    app = QtWidgets.QApplication.instance()
    if app is not None:
        return app, False
    return QtWidgets.QApplication([]), True


def _track_top_level_window(window: QtWidgets.QWidget):
    """Keep a Python reference to top-level Qt windows until they are destroyed."""
    _OPEN_QT_TOP_LEVEL_WINDOWS.append(window)

    def _release_window(*_args):
        try:
            _OPEN_QT_TOP_LEVEL_WINDOWS.remove(window)
        except ValueError:
            pass

    window.destroyed.connect(_release_window)
    return window


def _coalesce_seed_points(
    mask_seed_points: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None,
    mask_seed_point: tuple[int, int] | None = None,
) -> tuple[tuple[int, int], ...]:
    """Merge zero, one, or many seed points into a deduplicated tuple."""
    resolved_points = [
        (int(row), int(column))
        for row, column in (mask_seed_points or ())
    ]
    if mask_seed_point is not None:
        resolved_points.append((int(mask_seed_point[0]), int(mask_seed_point[1])))
    return tuple(dict.fromkeys(resolved_points))


def _toggle_seed_point(
    seed_points: list[tuple[int, int]],
    point: tuple[int, int],
) -> list[tuple[int, int]]:
    """Add a seed point when absent, or remove it when it is already selected."""
    normalized_point = (int(point[0]), int(point[1]))
    if normalized_point in seed_points:
        return [existing_point for existing_point in seed_points if existing_point != normalized_point]
    return [*seed_points, normalized_point]


def _format_seed_coordinate_summary(
    seed_points: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    *,
    max_points: int = 4,
) -> str:
    """Return a compact seed summary for labels and preview metadata."""
    normalized_points = tuple((int(row), int(column)) for row, column in seed_points)
    if not normalized_points:
        return "No seeds selected."
    preview_items = [f"x={column}, y={row}" for row, column in normalized_points[:max_points]]
    if len(normalized_points) > max_points:
        preview_items.append(f"+{len(normalized_points) - max_points} more")
    label = "seed" if len(normalized_points) == 1 else "seeds"
    return f"{len(normalized_points)} {label} selected ({'; '.join(preview_items)})."


def _format_interactive_seed_summary(
    seed_points: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    *,
    color_tolerance: int,
    region_offset: int,
) -> str:
    """Build a compact summary for interactive seed previews."""
    return (
        "Left click adds or toggles seed pixels. "
        f"{_format_seed_coordinate_summary(seed_points)} "
        f"Color tolerance: {int(color_tolerance)}. "
        f"Region offset: {int(region_offset):+d} px."
    )


class InteractiveSeedMaskControls(QtWidgets.QGroupBox):
    """Shared controls for tuning interactive seed masks."""

    values_changed = QtCore.Signal()
    clear_requested = QtCore.Signal()

    def __init__(self, title: str = "Interactive Seed Mask"):
        super().__init__(title)
        self._seed_points: tuple[tuple[int, int], ...] = ()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        instructions_label = QtWidgets.QLabel(
            "Left click on the image preview to add or toggle seed pixels. "
            "Positive region offset expands the masked area; negative values shrink it."
        )
        instructions_label.setWordWrap(True)
        layout.addWidget(instructions_label)

        self.seed_summary_label = QtWidgets.QLabel("No seeds selected.")
        self.seed_summary_label.setWordWrap(True)
        self.seed_summary_label.setStyleSheet("color: #cccccc;")
        layout.addWidget(self.seed_summary_label)

        self.color_tolerance_spinbox = self._build_spinbox_row(
            layout,
            label_text="Color Tolerance",
            minimum=0,
            maximum=MASK_COLOR_TOLERANCE_MAX,
            value=DEFAULT_SEED_MASK_COLOR_TOLERANCE,
        )
        self.region_offset_spinbox = self._build_spinbox_row(
            layout,
            label_text="Region Offset",
            minimum=MASK_REGION_OFFSET_MIN,
            maximum=MASK_REGION_OFFSET_MAX,
            value=0,
            suffix=" px",
        )

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        self.clear_seeds_button = QtWidgets.QPushButton("Clear Seeds")
        self.clear_seeds_button.clicked.connect(self.clear_requested)
        button_row.addWidget(self.clear_seeds_button)
        layout.addLayout(button_row)

        self.color_tolerance_spinbox.valueChanged.connect(self.values_changed)
        self.region_offset_spinbox.valueChanged.connect(self.values_changed)
        self.set_seed_points(())

    def _build_spinbox_row(
        self,
        layout: QtWidgets.QVBoxLayout,
        *,
        label_text: str,
        minimum: int,
        maximum: int,
        value: int,
        suffix: str = "",
    ) -> QtWidgets.QSpinBox:
        row_layout = QtWidgets.QGridLayout()
        row_layout.setHorizontalSpacing(8)
        row_layout.setVerticalSpacing(4)
        row_layout.setColumnStretch(1, 1)

        label = QtWidgets.QLabel(label_text)
        spinbox = QtWidgets.QSpinBox()
        spinbox.setRange(int(minimum), int(maximum))
        spinbox.setValue(int(value))
        spinbox.setSingleStep(1)
        spinbox.setAccelerated(True)
        spinbox.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        spinbox.setMinimumWidth(96)
        if suffix:
            spinbox.setSuffix(suffix)

        row_layout.addWidget(label, 0, 0)
        row_layout.addWidget(spinbox, 0, 2)
        layout.addLayout(row_layout)
        return spinbox

    def color_tolerance(self) -> int:
        return int(self.color_tolerance_spinbox.value())

    def region_offset(self) -> int:
        return int(self.region_offset_spinbox.value())

    def set_seed_points(self, seed_points: tuple[tuple[int, int], ...] | list[tuple[int, int]]):
        self._seed_points = tuple((int(row), int(column)) for row, column in seed_points)
        self.seed_summary_label.setText(_format_seed_coordinate_summary(self._seed_points))
        self.clear_seeds_button.setEnabled(bool(self._seed_points))


def _resolve_qt_mask_loading(
    parent: QtWidgets.QWidget | None,
    image_path: str,
    alpha_mask_path: str | None,
    mask_seed_points: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None,
    mask_seed_point: tuple[int, int] | None,
    *,
    mask_color_tolerance: int,
    mask_region_offset: int,
    prompt_user: bool,
) -> tuple[str | None, tuple[tuple[int, int], ...] | None, int, int, bool, bool]:
    """Resolve whether the editor should use an external mask, a clicked seed mask, or continue opaque."""
    resolved_seed_points = _coalesce_seed_points(mask_seed_points, mask_seed_point)
    if alpha_mask_path is not None:
        return alpha_mask_path, None, int(mask_color_tolerance), int(mask_region_offset), False, False
    if resolved_seed_points:
        return None, resolved_seed_points, int(mask_color_tolerance), int(mask_region_offset), False, False

    loaded_image = load_image_data(image_path)
    if not loaded_image.mask_prompt_required or not prompt_user:
        return None, None, int(mask_color_tolerance), int(mask_region_offset), False, False

    dialog = QtSeedMaskSelectionDialog(
        parent,
        image_path,
        image_warnings=loaded_image.image_warnings,
    )
    result = dialog.exec()
    if result == QtWidgets.QDialog.DialogCode.Accepted:
        if dialog.continue_without_mask_requested():
            return None, None, dialog.selected_mask_color_tolerance(), dialog.selected_mask_region_offset(), True, False
        return (
            None,
            dialog.selected_mask_seed_points(),
            dialog.selected_mask_color_tolerance(),
            dialog.selected_mask_region_offset(),
            True,
            False,
        )
    return None, None, int(mask_color_tolerance), int(mask_region_offset), True, True


class QtTargetImagePickerDialog(QtWidgets.QDialog):
    """Modal dialog for selecting one target image and which L/C/H curves should use it."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        *,
        initial_image_path: str | None = None,
        initial_mask_path: str | None = None,
        initial_mask_mode: str | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Load L/C/H Target")
        self.resize(1120, 760)
        self._mask_seed_points: list[tuple[int, int]] = []

        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(12)

        description_label = QtWidgets.QLabel(
            "Choose one target image, then tick which curves should import information from it."
        )
        description_label.setWordWrap(True)
        root_layout.addWidget(description_label)

        content_layout = QtWidgets.QHBoxLayout()
        content_layout.setSpacing(16)
        root_layout.addLayout(content_layout, 1)

        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(12)
        content_layout.addWidget(controls_widget, 1)

        path_layout = QtWidgets.QHBoxLayout()
        self.image_path_edit = QtWidgets.QLineEdit(initial_image_path or "")
        self.image_path_edit.setPlaceholderText("Target image path")
        self.browse_button = QtWidgets.QPushButton("Browse")
        path_layout.addWidget(self.image_path_edit, 1)
        path_layout.addWidget(self.browse_button)
        controls_layout.addLayout(path_layout)

        mask_mode_group = QtWidgets.QGroupBox("Mask Source")
        mask_mode_layout = QtWidgets.QVBoxLayout(mask_mode_group)
        mask_mode_layout.setSpacing(8)
        self.use_image_alpha_radio = QtWidgets.QRadioButton("Use image alpha / no extra mask")
        self.load_mask_radio = QtWidgets.QRadioButton("Load mask file")
        self.pick_region_radio = QtWidgets.QRadioButton("Pick connected region from image")
        mask_mode_layout.addWidget(self.use_image_alpha_radio)
        mask_mode_layout.addWidget(self.load_mask_radio)
        mask_mode_layout.addWidget(self.pick_region_radio)
        controls_layout.addWidget(mask_mode_group)

        mask_layout = QtWidgets.QHBoxLayout()
        self.mask_path_edit = QtWidgets.QLineEdit(initial_mask_path or "")
        self.mask_path_edit.setPlaceholderText("Optional target mask path")
        self.mask_browse_button = QtWidgets.QPushButton("Browse Mask")
        self.mask_clear_button = QtWidgets.QPushButton("Clear Mask")
        mask_layout.addWidget(self.mask_path_edit, 1)
        mask_layout.addWidget(self.mask_browse_button)
        mask_layout.addWidget(self.mask_clear_button)
        controls_layout.addLayout(mask_layout)

        self.seed_mask_controls = InteractiveSeedMaskControls()
        controls_layout.addWidget(self.seed_mask_controls)

        checkbox_layout = QtWidgets.QHBoxLayout()
        self.lightness_checkbox = QtWidgets.QCheckBox("Apply to L")
        self.chroma_checkbox = QtWidgets.QCheckBox("Apply to C")
        self.hue_checkbox = QtWidgets.QCheckBox("Apply to H")
        checkbox_layout.addWidget(self.lightness_checkbox)
        checkbox_layout.addWidget(self.chroma_checkbox)
        checkbox_layout.addWidget(self.hue_checkbox)
        checkbox_layout.addStretch(1)
        controls_layout.addLayout(checkbox_layout)
        controls_layout.addStretch(1)

        previews_widget = QtWidgets.QWidget()
        previews_layout = QtWidgets.QHBoxLayout(previews_widget)
        previews_layout.setContentsMargins(0, 0, 0, 0)
        previews_layout.setSpacing(12)
        content_layout.addWidget(previews_widget, 1)

        image_preview_group = QtWidgets.QGroupBox("Target Image Preview")
        image_preview_layout = QtWidgets.QVBoxLayout(image_preview_group)
        self.image_preview_label = ImagePreviewLabel("Target Image Preview")
        self.image_preview_label.setMinimumSize(320, 320)
        self.image_preview_info_label = QtWidgets.QLabel("Choose a target image to preview it.")
        self.image_preview_info_label.setWordWrap(True)
        image_preview_layout.addWidget(self.image_preview_label, 1)
        image_preview_layout.addWidget(self.image_preview_info_label)
        previews_layout.addWidget(image_preview_group, 1)

        mask_preview_group = QtWidgets.QGroupBox("Mask Preview")
        mask_preview_layout = QtWidgets.QVBoxLayout(mask_preview_group)
        self.mask_preview_label = ImagePreviewLabel("Mask Preview")
        self.mask_preview_label.setMinimumSize(320, 320)
        self.mask_preview_info_label = QtWidgets.QLabel("Choose a mask source to preview it.")
        self.mask_preview_info_label.setWordWrap(True)
        mask_preview_layout.addWidget(self.mask_preview_label, 1)
        mask_preview_layout.addWidget(self.mask_preview_info_label)
        previews_layout.addWidget(mask_preview_group, 1)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        root_layout.addWidget(self.button_box)

        self.browse_button.clicked.connect(self._browse_image)
        self.mask_browse_button.clicked.connect(self._browse_mask)
        self.mask_clear_button.clicked.connect(self._clear_mask)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.image_path_edit.textChanged.connect(self._on_image_path_changed)
        self.mask_path_edit.textChanged.connect(self._refresh_mask_preview)
        self.use_image_alpha_radio.toggled.connect(self._on_mask_mode_changed)
        self.load_mask_radio.toggled.connect(self._on_mask_mode_changed)
        self.pick_region_radio.toggled.connect(self._on_mask_mode_changed)
        self.image_preview_label.image_point_clicked.connect(self._handle_image_preview_click)
        self.seed_mask_controls.values_changed.connect(self._refresh_previews)
        self.seed_mask_controls.clear_requested.connect(self._clear_seed_points)

        self._set_initial_mask_mode(initial_mask_mode)
        self._sync_mask_controls()
        self._refresh_previews()

    def _set_initial_mask_mode(self, initial_mask_mode: str | None):
        mask_mode = initial_mask_mode or ("external" if self.mask_path_edit.text().strip() else "interactive-seed")
        if mask_mode == "external":
            self.load_mask_radio.setChecked(True)
        elif mask_mode == "interactive-seed":
            self.pick_region_radio.setChecked(True)
        else:
            self.use_image_alpha_radio.setChecked(True)

    def _mask_mode(self) -> str:
        if self.load_mask_radio.isChecked():
            return "external"
        if self.pick_region_radio.isChecked():
            return "interactive-seed"
        return "image-alpha"

    def _sync_mask_controls(self):
        mask_controls_enabled = self.load_mask_radio.isChecked()
        self.mask_path_edit.setEnabled(mask_controls_enabled)
        self.mask_browse_button.setEnabled(mask_controls_enabled)
        self.mask_clear_button.setEnabled(mask_controls_enabled)
        interactive_seed_enabled = self._mask_mode() == "interactive-seed"
        self.image_preview_label.set_click_enabled(interactive_seed_enabled)
        self.seed_mask_controls.setEnabled(interactive_seed_enabled)
        self.seed_mask_controls.set_seed_points(self._mask_seed_points)

    def _on_image_path_changed(self):
        self._mask_seed_points = []
        self._refresh_previews()

    def _on_mask_mode_changed(self):
        self._sync_mask_controls()
        self._refresh_previews()

    def _handle_image_preview_click(self, row: int, column: int):
        if self._mask_mode() != "interactive-seed":
            return
        self._mask_seed_points = _toggle_seed_point(self._mask_seed_points, (row, column))
        self._refresh_previews()

    def _clear_seed_points(self):
        self._mask_seed_points = []
        self._refresh_previews()

    def _clear_mask(self):
        self.mask_path_edit.clear()
        self.pick_region_radio.setChecked(True)

    def _refresh_previews(self):
        self._refresh_image_preview()
        self._refresh_mask_preview()

    def _refresh_image_preview(self):
        image_path = self.selected_image_path()
        if not image_path:
            _set_preview_panel_state(
                self.image_preview_label,
                self.image_preview_info_label,
                preview_rgb=None,
                placeholder_text="Target Image Preview",
                info_text="Choose a target image to preview it.",
            )
            return
        try:
            preview_rgb, image_shape = _load_image_preview_uint8(image_path)
        except Exception as exc:  # noqa: BLE001
            _set_preview_panel_state(
                self.image_preview_label,
                self.image_preview_info_label,
                preview_rgb=None,
                placeholder_text="Target Image Preview",
                info_text=f"Failed to preview image: {exc}",
                info_style="color: #ff9b9b;",
            )
            return
        if self._mask_mode() == "interactive-seed":
            summary = _format_interactive_seed_summary(
                self._mask_seed_points,
                color_tolerance=self.seed_mask_controls.color_tolerance(),
                region_offset=self.seed_mask_controls.region_offset(),
            )
        else:
            summary = "Previewing the selected target image."
        _set_preview_panel_state(
            self.image_preview_label,
            self.image_preview_info_label,
            preview_rgb=preview_rgb,
            placeholder_text="Target Image Preview",
            info_text=_format_image_preview_info(image_shape, summary=summary),
            source_image_shape=image_shape,
            marker_points=self._mask_seed_points if self._mask_mode() == "interactive-seed" else (),
        )

    def _refresh_mask_preview(self):
        image_path = self.selected_image_path()
        expected_shape = _load_optional_image_shape(image_path)
        mask_mode = self._mask_mode()
        if mask_mode == "external":
            mask_path = self.mask_path_edit.text().strip()
            if not mask_path:
                _set_preview_panel_state(
                    self.mask_preview_label,
                    self.mask_preview_info_label,
                    preview_rgb=None,
                    placeholder_text="Mask Preview",
                    info_text="Choose a mask file to preview it.",
                )
                return
            try:
                preview_rgb, mask_shape, valid_pixels = _load_mask_preview_uint8(mask_path)
                info_text, info_style = _format_mask_preview_info(
                    mask_shape,
                    valid_pixels,
                    summary="Previewing the selected external mask.",
                    expected_shape=expected_shape,
                )
            except Exception as exc:  # noqa: BLE001
                _set_preview_panel_state(
                    self.mask_preview_label,
                    self.mask_preview_info_label,
                    preview_rgb=None,
                    placeholder_text="Mask Preview",
                    info_text=f"Failed to preview mask: {exc}",
                    info_style="color: #ff9b9b;",
                )
                return
        elif mask_mode == "interactive-seed":
            if not image_path:
                _set_preview_panel_state(
                    self.mask_preview_label,
                    self.mask_preview_info_label,
                    preview_rgb=None,
                    placeholder_text="Mask Preview",
                    info_text="Choose a target image, then click one or more pixels in the image preview.",
                )
                return
            if not self._mask_seed_points:
                _set_preview_panel_state(
                    self.mask_preview_label,
                    self.mask_preview_info_label,
                    preview_rgb=None,
                    placeholder_text="Mask Preview",
                    info_text="Click one or more pixels in the target image preview to build a connected-region mask.",
                )
                return
            try:
                preview_rgb, mask_shape, valid_pixels, summary = _load_seeded_mask_preview_uint8(
                    image_path,
                    self._mask_seed_points,
                    color_tolerance=self.seed_mask_controls.color_tolerance(),
                    region_offset=self.seed_mask_controls.region_offset(),
                )
                info_text, info_style = _format_mask_preview_info(
                    mask_shape,
                    valid_pixels,
                    summary=summary,
                    expected_shape=expected_shape,
                )
            except Exception as exc:  # noqa: BLE001
                _set_preview_panel_state(
                    self.mask_preview_label,
                    self.mask_preview_info_label,
                    preview_rgb=None,
                    placeholder_text="Mask Preview",
                    info_text=f"Connected-region preview failed: {exc}",
                    info_style="color: #ff9b9b;",
                )
                return
        else:
            if not image_path:
                _set_preview_panel_state(
                    self.mask_preview_label,
                    self.mask_preview_info_label,
                    preview_rgb=None,
                    placeholder_text="Mask Preview",
                    info_text="Choose a target image to preview its embedded or implicit mask.",
                )
                return
            try:
                preview_rgb, mask_shape, valid_pixels, summary = _load_embedded_or_implicit_mask_preview_uint8(image_path)
                info_text, info_style = _format_mask_preview_info(
                    mask_shape,
                    valid_pixels,
                    summary=summary,
                    expected_shape=expected_shape,
                )
            except Exception as exc:  # noqa: BLE001
                _set_preview_panel_state(
                    self.mask_preview_label,
                    self.mask_preview_info_label,
                    preview_rgb=None,
                    placeholder_text="Mask Preview",
                    info_text=f"Failed to preview mask source: {exc}",
                    info_style="color: #ff9b9b;",
                )
                return

        _set_preview_panel_state(
            self.mask_preview_label,
            self.mask_preview_info_label,
            preview_rgb=preview_rgb,
            placeholder_text="Mask Preview",
            info_text=info_text,
            info_style=info_style,
        )

    def _browse_image(self):
        selected_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select target image",
            os.path.dirname(self.image_path_edit.text().strip() or os.getcwd()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if selected_path:
            self.image_path_edit.setText(selected_path)

    def _browse_mask(self):
        selected_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select target mask image",
            os.path.dirname(self.mask_path_edit.text().strip() or self.image_path_edit.text().strip() or os.getcwd()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if selected_path:
            self.load_mask_radio.setChecked(True)
            self.mask_path_edit.setText(selected_path)

    def selected_image_path(self) -> str | None:
        value = self.image_path_edit.text().strip()
        return value or None

    def selected_mask_path(self) -> str | None:
        if not self.load_mask_radio.isChecked():
            return None
        value = self.mask_path_edit.text().strip()
        return value or None

    def selected_mask_seed_point(self) -> tuple[int, int] | None:
        if self._mask_mode() != "interactive-seed":
            return None
        return self._mask_seed_points[-1] if self._mask_seed_points else None

    def selected_mask_seed_points(self) -> tuple[tuple[int, int], ...] | None:
        if self._mask_mode() != "interactive-seed":
            return None
        return tuple(self._mask_seed_points)

    def selected_mask_color_tolerance(self) -> int:
        return self.seed_mask_controls.color_tolerance()

    def selected_mask_region_offset(self) -> int:
        return self.seed_mask_controls.region_offset()

    def selected_curve_flags(self) -> tuple[bool, bool, bool]:
        return (
            self.lightness_checkbox.isChecked(),
            self.chroma_checkbox.isChecked(),
            self.hue_checkbox.isChecked(),
        )

    def accept(self):
        image_path = self.selected_image_path()
        mask_path = self.selected_mask_path()
        apply_lightness, apply_chroma, apply_hue = self.selected_curve_flags()
        if image_path is None:
            QtWidgets.QMessageBox.critical(self, "Target Image Required", "Please choose a target image first.")
            return
        if self.load_mask_radio.isChecked() and mask_path is None:
            QtWidgets.QMessageBox.critical(self, "Target Mask Required", "Please choose a target mask first.")
            return
        if self._mask_mode() == "interactive-seed" and not self._mask_seed_points:
            QtWidgets.QMessageBox.critical(
                self,
                "Seed Pixel Required",
                "Please click one or more pixels in the target image preview first.",
            )
            return
        if mask_path is not None:
            image_shape = _load_optional_image_shape(image_path)
            mask_shape = _load_optional_image_shape(mask_path)
            if image_shape is not None and mask_shape is not None and tuple(image_shape) != tuple(mask_shape):
                QtWidgets.QMessageBox.critical(
                    self,
                    "Target Mask Size Mismatch",
                    "Target mask size must match the selected target image.",
                )
                return
        if not any((apply_lightness, apply_chroma, apply_hue)):
            QtWidgets.QMessageBox.critical(self, "Target Curves Required", "Select at least one of L, C, or H.")
            return
        super().accept()


class ImagePreviewLabel(QtWidgets.QLabel):
    """QLabel that keeps an RGB preview scaled to the available size."""

    image_point_clicked = QtCore.Signal(int, int)

    def __init__(self, title: str):
        super().__init__()
        self._title = title
        self._pixmap: QtGui.QPixmap | None = None
        self._preview_shape: tuple[int, int] | None = None
        self._source_shape: tuple[int, int] | None = None
        self._marker_points: tuple[tuple[int, int], ...] = ()
        self._click_enabled = False
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 320)
        self.setStyleSheet("background-color: #111; border: 1px solid #333;")
        self.setText(title)

    def set_rgb_uint8(self, rgb_uint8: np.ndarray, *, source_image_shape: tuple[int, int] | None = None):
        """Update the displayed RGB image."""
        self._pixmap = QtGui.QPixmap.fromImage(_rgb_uint8_to_qimage(rgb_uint8))
        self._preview_shape = tuple(int(value) for value in rgb_uint8.shape[:2])
        self._source_shape = source_image_shape or self._preview_shape
        self._refresh_pixmap()
        self._refresh_cursor()

    def clear_preview(self, text: str | None = None):
        """Reset the preview back to a text placeholder."""
        self._pixmap = None
        self._preview_shape = None
        self._source_shape = None
        self._marker_points = ()
        self.clear()
        self.setText(text or self._title)
        self._refresh_cursor()

    def set_marker_points(self, marker_points: tuple[tuple[int, int], ...] | list[tuple[int, int]]):
        """Overlay source-image marker points on top of the preview."""
        self._marker_points = tuple((int(row), int(column)) for row, column in marker_points)
        self._refresh_pixmap()

    def set_click_enabled(self, enabled: bool):
        """Enable or disable image-coordinate click reporting."""
        self._click_enabled = bool(enabled)
        self._refresh_cursor()

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        super().mousePressEvent(event)
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        image_point = self._map_event_to_image_point(event.position())
        if image_point is None:
            return
        self.image_point_clicked.emit(image_point[0], image_point[1])

    def _refresh_pixmap(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        if self._marker_points and self._source_shape is not None:
            scaled = scaled.copy()
            source_height, source_width = self._source_shape
            painter = QtGui.QPainter(scaled)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            marker_radius = max(4.0, min(float(scaled.width()), float(scaled.height())) / 45.0)
            pen = QtGui.QPen(SEED_MARKER_OUTLINE_COLOR)
            pen.setWidthF(max(1.5, marker_radius / 3.0))
            painter.setPen(pen)
            painter.setBrush(SEED_MARKER_FILL_COLOR)
            for row, column in self._marker_points:
                x_value = (float(column) + 0.5) * float(scaled.width()) / max(1.0, float(source_width))
                y_value = (float(row) + 0.5) * float(scaled.height()) / max(1.0, float(source_height))
                painter.drawEllipse(QtCore.QPointF(x_value, y_value), marker_radius, marker_radius)
            painter.end()
        self.setPixmap(scaled)

    def _refresh_cursor(self):
        if self._click_enabled and self._pixmap is not None:
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            return
        self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

    def _map_event_to_image_point(self, position: QtCore.QPointF) -> tuple[int, int] | None:
        if not self._click_enabled or self._pixmap is None or self._preview_shape is None or self._source_shape is None:
            return None
        scaled = self._pixmap.scaled(
            self.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        x_offset = (self.width() - scaled.width()) / 2.0
        y_offset = (self.height() - scaled.height()) / 2.0
        x_value = float(position.x()) - x_offset
        y_value = float(position.y()) - y_offset
        if x_value < 0.0 or y_value < 0.0 or x_value >= float(scaled.width()) or y_value >= float(scaled.height()):
            return None
        preview_height, preview_width = self._preview_shape
        source_height, source_width = self._source_shape
        preview_column = min(preview_width - 1, int(x_value * preview_width / max(1, scaled.width())))
        preview_row = min(preview_height - 1, int(y_value * preview_height / max(1, scaled.height())))
        source_column = min(source_width - 1, int(preview_column * source_width / max(1, preview_width)))
        source_row = min(source_height - 1, int(preview_row * source_height / max(1, preview_height)))
        return source_row, source_column


def _load_optional_image_shape(image_path: str | None) -> tuple[int, int] | None:
    """Return the image shape as (height, width) when the file exists and is readable."""
    if not image_path:
        return None
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception:  # noqa: BLE001
        return None
    return height, width


def _load_image_preview_uint8(image_path: str, *, max_side: int = 640) -> tuple[np.ndarray, tuple[int, int]]:
    """Load an image file and return a small RGB preview plus its original shape."""
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        preview_image = rgb_image.copy()
        preview_image.thumbnail((int(max_side), int(max_side)), PIL_PREVIEW_RESAMPLE)
    return np.asarray(preview_image, dtype=np.uint8), (height, width)


def _mask_to_preview_uint8(valid_mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int]:
    """Convert a boolean valid-mask into a black/white RGB preview image."""
    valid_mask = np.asarray(valid_mask, dtype=bool)
    preview_gray = np.where(valid_mask, 255, 0).astype(np.uint8)
    preview_rgb = np.repeat(preview_gray[:, :, None], 3, axis=2)
    return preview_rgb, tuple(int(value) for value in valid_mask.shape), int(np.count_nonzero(valid_mask))


def _load_mask_preview_uint8(mask_path: str) -> tuple[np.ndarray, tuple[int, int], int]:
    """Load a mask file and convert it into a black/white RGB preview."""
    with Image.open(mask_path) as image:
        mask_array = np.asarray(image.convert("L"), dtype=np.uint8)
    valid_mask = np.asarray(mask_array > 0, dtype=bool)
    return _mask_to_preview_uint8(valid_mask)


def _load_embedded_or_implicit_mask_preview_uint8(
    image_path: str,
) -> tuple[np.ndarray, tuple[int, int], int, str]:
    """Load the input image's embedded alpha preview, or a full-image fallback when no alpha exists."""
    with Image.open(image_path) as image:
        width, height = image.size
        if "A" in image.getbands():
            alpha_plane = np.asarray(image.getchannel("A"), dtype=np.uint8)
            preview_rgb, mask_shape, valid_pixels = _mask_to_preview_uint8(alpha_plane > 0)
            if np.all(alpha_plane == 255):
                return preview_rgb, mask_shape, valid_pixels, "Embedded alpha is fully opaque."
            return preview_rgb, mask_shape, valid_pixels, "Previewing embedded alpha coverage."
    implicit_mask = np.ones((height, width), dtype=bool)
    preview_rgb, mask_shape, valid_pixels = _mask_to_preview_uint8(implicit_mask)
    return preview_rgb, mask_shape, valid_pixels, "No embedded alpha; previewing full-image coverage."


def _load_seeded_mask_preview_uint8(
    image_path: str,
    seed_points: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    *,
    color_tolerance: int,
    region_offset: int,
) -> tuple[np.ndarray, tuple[int, int], int, str]:
    """Generate a preview for a connected-region mask grown from one or more user-selected seed pixels."""
    with Image.open(image_path) as image:
        rgb_float = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    resolved_seed_points = tuple((int(row), int(column)) for row, column in seed_points)
    valid_mask = detect_seeded_valid_mask(
        rgb_float,
        resolved_seed_points,
        color_tolerance=int(color_tolerance),
        region_offset_pixels=int(region_offset),
    )
    preview_rgb, mask_shape, valid_pixels = _mask_to_preview_uint8(valid_mask)
    seed_label = "seed pixel" if len(resolved_seed_points) == 1 else "seed pixels"
    return (
        preview_rgb,
        mask_shape,
        valid_pixels,
        "Previewing the connected-region mask from {} {}. Color tolerance: {}. Region offset: {:+d} px.".format(
            len(resolved_seed_points),
            seed_label,
            int(color_tolerance),
            int(region_offset),
        ),
    )


def _set_preview_panel_state(
    preview_label: ImagePreviewLabel,
    info_label: QtWidgets.QLabel,
    *,
    preview_rgb: np.ndarray | None,
    placeholder_text: str,
    info_text: str,
    info_style: str = "color: #cccccc;",
    source_image_shape: tuple[int, int] | None = None,
    marker_points: tuple[tuple[int, int], ...] | list[tuple[int, int]] = (),
):
    """Update a preview label plus its companion info label."""
    if preview_rgb is None:
        preview_label.clear_preview(placeholder_text)
    else:
        preview_label.set_rgb_uint8(preview_rgb, source_image_shape=source_image_shape)
        preview_label.set_marker_points(marker_points)
    info_label.setText(info_text)
    info_label.setStyleSheet(info_style)


def _format_image_preview_info(image_shape: tuple[int, int], *, summary: str | None = None) -> str:
    """Build a compact info string for an RGB image preview."""
    info_lines = []
    if summary:
        info_lines.append(summary)
    info_lines.append("Shape: {} x {}".format(image_shape[1], image_shape[0]))
    return "\n".join(info_lines)


def _format_mask_preview_info(
    mask_shape: tuple[int, int],
    valid_pixels: int,
    *,
    summary: str,
    expected_shape: tuple[int, int] | None = None,
) -> tuple[str, str]:
    """Build a compact info string plus style for a mask preview panel."""
    total_pixels = max(1, int(mask_shape[0] * mask_shape[1]))
    coverage = 100.0 * float(valid_pixels) / float(total_pixels)
    info_lines = [
        summary,
        "Shape: {} x {}".format(mask_shape[1], mask_shape[0]),
        "Valid pixels: {} / {} ({:.1f}%)".format(valid_pixels, total_pixels, coverage),
    ]
    info_style = "color: #cccccc;"
    if expected_shape is not None:
        if tuple(expected_shape) == tuple(mask_shape):
            info_lines.append("Matches selected image size.")
            info_style = "color: #8fe388;"
        else:
            info_lines.append(
                "Does not match selected image size: expected {} x {}.".format(
                    expected_shape[1],
                    expected_shape[0],
                )
            )
            info_style = "color: #ff9b9b;"
    return "\n".join(info_lines), info_style


class QtMaskPreviewDialog(QtWidgets.QDialog):
    """Modal dialog that previews the binary shape of a selected mask."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        mask_path: str,
        *,
        expected_shape: tuple[int, int] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Mask Preview")
        self.resize(520, 620)

        preview_rgb, mask_shape, valid_pixels = _load_mask_preview_uint8(mask_path)
        total_pixels = max(1, mask_shape[0] * mask_shape[1])
        coverage = 100.0 * valid_pixels / total_pixels

        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(12)

        path_label = QtWidgets.QLabel(mask_path)
        path_label.setWordWrap(True)
        root_layout.addWidget(path_label)

        self.preview_label = ImagePreviewLabel("Mask Preview")
        self.preview_label.setMinimumSize(420, 420)
        self.preview_label.set_rgb_uint8(preview_rgb)
        root_layout.addWidget(self.preview_label, 1)

        self.shape_label = QtWidgets.QLabel(
            "Shape: {} x {}    Valid pixels: {} / {} ({:.1f}%)".format(
                mask_shape[1],
                mask_shape[0],
                valid_pixels,
                total_pixels,
                coverage,
            )
        )
        root_layout.addWidget(self.shape_label)

        if expected_shape is None:
            comparison_text = "No reference image selected yet; size check will happen when the image is loaded."
            comparison_style = "color: #cccccc;"
        elif tuple(expected_shape) == tuple(mask_shape):
            comparison_text = "Mask shape matches the selected image."
            comparison_style = "color: #8fe388;"
        else:
            comparison_text = "Mask shape does not match the selected image: expected {} x {}, got {} x {}.".format(
                expected_shape[1],
                expected_shape[0],
                mask_shape[1],
                mask_shape[0],
            )
            comparison_style = "color: #ff9b9b;"
        self.comparison_label = QtWidgets.QLabel(comparison_text)
        self.comparison_label.setWordWrap(True)
        self.comparison_label.setStyleSheet(comparison_style)
        root_layout.addWidget(self.comparison_label)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        root_layout.addWidget(self.button_box)


def _select_mask_path_with_preview(
    parent: QtWidgets.QWidget | None,
    *,
    title: str,
    initial_directory: str,
    expected_shape: tuple[int, int] | None = None,
) -> str | None:
    """Let the user choose a mask file, then confirm it from a binary-shape preview dialog."""
    selected_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent,
        title,
        initial_directory,
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
    )
    if not selected_path:
        return None
    preview_dialog = QtMaskPreviewDialog(parent, selected_path, expected_shape=expected_shape)
    if preview_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    return selected_path


class QtSeedMaskSelectionDialog(QtWidgets.QDialog):
    """Modal dialog that lets the user click seed pixels and preview the connected-region mask."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        image_path: str,
        *,
        image_warnings: tuple[str, ...] = (),
    ):
        super().__init__(parent)
        self.setWindowTitle("Select Mask Seed")
        self.resize(980, 720)
        self._image_path = image_path
        self._mask_seed_points: list[tuple[int, int]] = []
        self._continue_without_mask = False

        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(12)

        instructions = list(image_warnings)
        instructions.append(
            "Click one or more pixels in the image preview. Each connected region grown from those pixels will be masked out."
        )
        instructions.append("Use 'Continue Without Extra Mask' if you want to keep the whole image valid.")
        description_label = QtWidgets.QLabel("\n\n".join(instructions))
        description_label.setWordWrap(True)
        root_layout.addWidget(description_label)

        self.seed_mask_controls = InteractiveSeedMaskControls()
        root_layout.addWidget(self.seed_mask_controls)

        previews_layout = QtWidgets.QHBoxLayout()
        previews_layout.setSpacing(12)
        root_layout.addLayout(previews_layout, 1)

        image_group = QtWidgets.QGroupBox("Input Image Preview")
        image_layout = QtWidgets.QVBoxLayout(image_group)
        self.image_preview_label = ImagePreviewLabel("Input Image Preview")
        self.image_preview_label.setMinimumSize(360, 360)
        self.image_preview_label.set_click_enabled(True)
        self.image_preview_info_label = QtWidgets.QLabel("Loading input image preview...")
        self.image_preview_info_label.setWordWrap(True)
        image_layout.addWidget(self.image_preview_label, 1)
        image_layout.addWidget(self.image_preview_info_label)
        previews_layout.addWidget(image_group, 1)

        mask_group = QtWidgets.QGroupBox("Mask Preview")
        mask_layout = QtWidgets.QVBoxLayout(mask_group)
        self.mask_preview_label = ImagePreviewLabel("Mask Preview")
        self.mask_preview_label.setMinimumSize(360, 360)
        self.mask_preview_info_label = QtWidgets.QLabel("Click one pixel in the image preview to build a connected-region mask.")
        self.mask_preview_info_label.setWordWrap(True)
        mask_layout.addWidget(self.mask_preview_label, 1)
        mask_layout.addWidget(self.mask_preview_info_label)
        previews_layout.addWidget(mask_group, 1)

        self.button_box = QtWidgets.QDialogButtonBox()
        self.use_seed_button = self.button_box.addButton(
            "Use Selected Region",
            QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole,
        )
        self.continue_button = self.button_box.addButton(
            "Continue Without Extra Mask",
            QtWidgets.QDialogButtonBox.ButtonRole.ActionRole,
        )
        self.cancel_button = self.button_box.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        root_layout.addWidget(self.button_box)

        self.use_seed_button.clicked.connect(self.accept)
        self.continue_button.clicked.connect(self._continue_without_mask_clicked)
        self.cancel_button.clicked.connect(self.reject)
        self.image_preview_label.image_point_clicked.connect(self._handle_image_preview_click)
        self.seed_mask_controls.values_changed.connect(self._refresh_previews)
        self.seed_mask_controls.clear_requested.connect(self._clear_seed_points)

        self._refresh_previews()

    def selected_mask_seed_point(self) -> tuple[int, int] | None:
        return self._mask_seed_points[-1] if self._mask_seed_points else None

    def selected_mask_seed_points(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._mask_seed_points)

    def selected_mask_color_tolerance(self) -> int:
        return self.seed_mask_controls.color_tolerance()

    def selected_mask_region_offset(self) -> int:
        return self.seed_mask_controls.region_offset()

    def continue_without_mask_requested(self) -> bool:
        return self._continue_without_mask

    def accept(self):
        if self._continue_without_mask:
            super().accept()
            return
        if not self._mask_seed_points:
            QtWidgets.QMessageBox.critical(
                self,
                "Seed Pixel Required",
                "Please click one or more pixels in the image preview first.",
            )
            return
        super().accept()

    def _continue_without_mask_clicked(self):
        self._continue_without_mask = True
        super().accept()

    def _handle_image_preview_click(self, row: int, column: int):
        self._continue_without_mask = False
        self._mask_seed_points = _toggle_seed_point(self._mask_seed_points, (row, column))
        self._refresh_previews()

    def _clear_seed_points(self):
        self._continue_without_mask = False
        self._mask_seed_points = []
        self._refresh_previews()

    def _refresh_previews(self):
        self._refresh_image_preview()
        self._refresh_mask_preview()

    def _refresh_image_preview(self):
        try:
            preview_rgb, image_shape = _load_image_preview_uint8(self._image_path)
        except Exception as exc:  # noqa: BLE001
            _set_preview_panel_state(
                self.image_preview_label,
                self.image_preview_info_label,
                preview_rgb=None,
                placeholder_text="Input Image Preview",
                info_text=f"Failed to preview input image: {exc}",
                info_style="color: #ff9b9b;",
            )
            return
        summary = "Click one pixel to remove its connected region from the mask."
        summary = _format_interactive_seed_summary(
            self._mask_seed_points,
            color_tolerance=self.seed_mask_controls.color_tolerance(),
            region_offset=self.seed_mask_controls.region_offset(),
        )
        _set_preview_panel_state(
            self.image_preview_label,
            self.image_preview_info_label,
            preview_rgb=preview_rgb,
            placeholder_text="Input Image Preview",
            info_text=_format_image_preview_info(image_shape, summary=summary),
            source_image_shape=image_shape,
            marker_points=self._mask_seed_points,
        )
        self.seed_mask_controls.set_seed_points(self._mask_seed_points)

    def _refresh_mask_preview(self):
        expected_shape = _load_optional_image_shape(self._image_path)
        if not self._mask_seed_points:
            _set_preview_panel_state(
                self.mask_preview_label,
                self.mask_preview_info_label,
                preview_rgb=None,
                placeholder_text="Mask Preview",
                info_text="Click one or more pixels in the image preview to build a connected-region mask.",
            )
            return
        try:
            preview_rgb, mask_shape, valid_pixels, summary = _load_seeded_mask_preview_uint8(
                self._image_path,
                self._mask_seed_points,
                color_tolerance=self.seed_mask_controls.color_tolerance(),
                region_offset=self.seed_mask_controls.region_offset(),
            )
            info_text, info_style = _format_mask_preview_info(
                mask_shape,
                valid_pixels,
                summary=summary,
                expected_shape=expected_shape,
            )
        except Exception as exc:  # noqa: BLE001
            _set_preview_panel_state(
                self.mask_preview_label,
                self.mask_preview_info_label,
                preview_rgb=None,
                placeholder_text="Mask Preview",
                info_text=f"Connected-region preview failed: {exc}",
                info_style="color: #ff9b9b;",
            )
            return
        _set_preview_panel_state(
            self.mask_preview_label,
            self.mask_preview_info_label,
            preview_rgb=preview_rgb,
            placeholder_text="Mask Preview",
            info_text=info_text,
            info_style=info_style,
        )


class DraggableCurveGraph(pg.GraphItem):
    """A draggable control-point polyline with ordered x positions."""

    sigControlPointsChanged = QtCore.Signal(object)
    sigControlPointPositionsChanged = QtCore.Signal(object)

    _X_DRAG_MARGIN = 1e-3

    def __init__(self, fixed_x: np.ndarray, y_min: float, y_max: float, pen, brush):
        super().__init__()
        self._y_min = float(y_min)
        self._y_max = float(y_max)
        self._pen = pen
        self._brush = brush
        self._adj = np.empty((0, 2), dtype=np.int32)
        self._point_data = np.empty(0, dtype=np.int32)
        self._positions = np.empty((0, 2), dtype=np.float64)
        self._drag_index: int | None = None
        self._drag_offset = pg.Point(0.0, 0.0)
        self.set_points(fixed_x, np.zeros_like(np.asarray(fixed_x, dtype=np.float64)))

    def set_points(self, x_values: np.ndarray, y_values: np.ndarray):
        """Replace the draggable control points."""
        x_values = np.asarray(x_values, dtype=np.float64)
        y_values = np.asarray(y_values, dtype=np.float64)
        if x_values.ndim != 1 or y_values.ndim != 1 or x_values.size != y_values.size:
            raise ValueError("control points must be parallel 1D x/y arrays")
        if x_values.size < 2:
            raise ValueError("at least two control points are required")

        sort_indices = np.argsort(x_values, kind="stable")
        x_values = np.clip(x_values[sort_indices], 0.0, 1.0)
        y_values = y_values[sort_indices]
        self._adj = _build_polyline_adjacency(x_values.size)
        self._point_data = np.arange(x_values.size, dtype=np.int32)
        self._positions = np.column_stack([x_values, y_values])
        self._update_graph()

    def set_y_values(self, y_values: np.ndarray):
        """Replace the draggable control-point y values."""
        self.set_points(self._positions[:, 0], y_values)

    def positions(self) -> np.ndarray:
        """Return a copy of the current control-point positions."""
        return np.array(self._positions, copy=True)

    def set_y_range(self, y_min: float, y_max: float):
        """Update the vertical clamp range used during dragging."""
        self._y_min = float(y_min)
        self._y_max = float(y_max)

    def _update_graph(self):
        pg.GraphItem.setData(
            self,
            pos=self._positions,
            adj=self._adj,
            data=self._point_data,
            pen=self._pen,
            symbol="o",
            size=12,
            symbolBrush=self._brush,
            symbolPen=pg.mkPen("#ffffff", width=1.2),
            pxMode=True,
        )

    def mouseDragEvent(self, event):
        """Drag a control point while preserving sorted x order."""
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            event.ignore()
            return

        if event.isStart():
            points = self.scatter.pointsAt(event.buttonDownPos())
            if not points:
                event.ignore()
                return
            self._drag_index = int(points[0].data())
            self._drag_offset = self._positions[self._drag_index] - event.buttonDownPos()
        elif event.isFinish():
            self._drag_index = None
            return
        elif self._drag_index is None:
            event.ignore()
            return

        new_position = event.pos() + self._drag_offset
        new_x = self._constrain_drag_x(self._drag_index, float(new_position.x()))
        new_y = float(np.clip(new_position.y(), self._y_min, self._y_max))
        self._positions[self._drag_index, 0] = new_x
        self._positions[self._drag_index, 1] = new_y
        self._update_graph()
        self.sigControlPointsChanged.emit(self._positions[:, 1].copy())
        self.sigControlPointPositionsChanged.emit(self.positions())
        event.accept()

    def _constrain_drag_x(self, index: int, proposed_x: float) -> float:
        """Clamp one point's x so interior points never cross their neighbors."""
        point_count = int(self._positions.shape[0])
        if point_count < 2:
            return float(np.clip(proposed_x, 0.0, 1.0))
        if index <= 0:
            return float(self._positions[0, 0])
        if index >= point_count - 1:
            return float(self._positions[-1, 0])

        left_x = float(self._positions[index - 1, 0]) + self._X_DRAG_MARGIN
        right_x = float(self._positions[index + 1, 0]) - self._X_DRAG_MARGIN
        if right_x <= left_x:
            midpoint = 0.5 * (left_x + right_x)
            return float(np.clip(midpoint, 0.0, 1.0))
        return float(np.clip(proposed_x, left_x, right_x))


class CurvePlotWidget(pg.PlotWidget):
    """Single-channel plot with background, line, histogram, and draggable points."""

    def __init__(
        self,
        *,
        title: str,
        x_label: str,
        y_label: str,
        y_range: tuple[float, float],
        fixed_x: np.ndarray,
        line_pen,
        point_brush,
    ):
        super().__init__()
        self._y_min, self._y_max = y_range
        self.getPlotItem().setTitle(title)
        self.getPlotItem().setLabel("bottom", x_label)
        self.getPlotItem().setLabel("left", y_label)
        self.getPlotItem().showGrid(x=True, y=True, alpha=0.15)
        self.getPlotItem().hideButtons()
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.setLimits(xMin=0.0, xMax=1.0, yMin=self._y_min, yMax=self._y_max)
        self.setXRange(0.0, 1.0, padding=0.0)
        self.setYRange(self._y_min, self._y_max, padding=0.0)
        self.getViewBox().setDefaultPadding(0.0)

        self._background_item = pg.ImageItem(axisOrder="row-major")
        self._background_item.setZValue(-20)
        self.addItem(self._background_item)

        self._hist_item = pg.PlotDataItem(
            pen=pg.mkPen("#ffffff", width=1.0),
            brush=pg.mkBrush(255, 255, 255, 70),
            fillLevel=0.0,
        )
        self._hist_item.setZValue(-5)
        self.addItem(self._hist_item)

        self._reference_hist_item = pg.PlotDataItem(
            pen=pg.mkPen("#7cf27c", width=1.5),
        )
        self._reference_hist_item.setZValue(-4)
        self.addItem(self._reference_hist_item)

        self._default_curve_item = pg.PlotCurveItem(pen=pg.mkPen("#b0b0b0", width=1.0, style=QtCore.Qt.PenStyle.DashLine))
        self._default_curve_item.setZValue(5)
        self.addItem(self._default_curve_item)

        self._curve_item = pg.PlotCurveItem(pen=line_pen)
        self._curve_item.setZValue(10)
        self.addItem(self._curve_item)

        self._control_item = DraggableCurveGraph(fixed_x, self._y_min, self._y_max, line_pen, point_brush)
        self._control_item.setZValue(15)
        self.addItem(self._control_item)

    @property
    def control_item(self) -> DraggableCurveGraph:
        """Return the draggable control point graph."""
        return self._control_item

    def set_background_rgb(self, rgb_uint8: np.ndarray):
        """Set the plot background to an RGB image."""
        self._background_item.setImage(np.ascontiguousarray(rgb_uint8), autoLevels=False)
        self._background_item.setRect(QtCore.QRectF(0.0, self._y_min, 1.0, self._y_max - self._y_min))

    def set_curve_line(self, y_values: np.ndarray):
        """Set the dense curve line."""
        self._curve_item.setData(CURVE_X_DENSE, y_values)

    def set_default_line(self, y_values: np.ndarray | None):
        """Set the default reference line."""
        if y_values is None:
            self._default_curve_item.setData([], [])
            return
        self._default_curve_item.setData(CURVE_X_DENSE, y_values)

    def set_control_points(self, x_values: np.ndarray, y_values: np.ndarray):
        """Set the draggable control-point positions."""
        self._control_item.set_points(x_values, y_values)

    def set_y_range(self, y_range: tuple[float, float]):
        """Update the view range and drag clamp range."""
        self._y_min, self._y_max = y_range
        self.setLimits(xMin=0.0, xMax=1.0, yMin=self._y_min, yMax=self._y_max)
        self.setYRange(self._y_min, self._y_max, padding=0.0)
        self._background_item.setRect(QtCore.QRectF(0.0, self._y_min, 1.0, self._y_max - self._y_min))
        self._control_item.set_y_range(self._y_min, self._y_max)

    def set_histogram(self, x_values: np.ndarray | None, y_values: np.ndarray | None):
        """Update the histogram overlay."""
        if x_values is None or y_values is None:
            self._hist_item.setData([], [])
            return
        self._hist_item.setData(x_values, y_values)

    def set_reference_histogram(self, x_values: np.ndarray | None, y_values: np.ndarray | None):
        """Update an optional reference histogram overlay."""
        if x_values is None or y_values is None:
            self._reference_hist_item.setData([], [])
            return
        self._reference_hist_item.setData(x_values, y_values)


class QtOklchCurveEditorWindow(QtWidgets.QMainWindow):
    """Qt MVP window for the Oklch state-curve editor."""

    def __init__(
        self,
        image_path: str,
        rgb_float: np.ndarray,
        oklch_float: np.ndarray,
        valid_mask: np.ndarray,
        base_model: OklchCurveModel,
        *,
        initial_curve_overrides: dict | None = None,
        dither_strength: float = DITHER_STRENGTH,
        curve_output_path: str | None = None,
    ):
        super().__init__()
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.image_path = image_path
        self.rgb_float = rgb_float
        self.oklch_float = oklch_float
        self.valid_mask = valid_mask
        self.base_model = base_model
        self.dither_strength = dither_strength
        self.curve_output_path = curve_output_path or self._default_curve_output_path()
        self.output_image_path = self._default_output_image_path()
        self.source_lightness_samples = self.oklch_float[self.valid_mask, 0]
        self.target_curve_image_paths: dict[str, str] = {}
        self.target_curve_mask_paths: dict[str, str | None] = {}
        self._lightness_reference_histogram: tuple[np.ndarray, np.ndarray] | None = None
        self._last_target_image_dialog_path = self.image_path
        self._last_target_mask_dialog_path: str | None = None
        self._last_target_mask_mode = "interactive-seed"
        self._default_curve_lines = self._build_default_curve_lines()
        self.hue_display_start = float(HUE_DISPLAY_START_MIN)
        self.hue_display_start_slider: QtWidgets.QSlider | None = None
        self.hue_display_start_value_label: QtWidgets.QLabel | None = None

        self._build_preview_inputs()
        self._initialize_controls(initial_curve_overrides or {})
        self._last_state_curves = None

        self._build_ui()
        self._render_preview()

    def _default_curve_output_path(self) -> str:
        stem, _ = os.path.splitext(self.image_path)
        return f"{stem}_state_curves.json"

    def _default_output_image_path(self) -> str:
        stem, _ = os.path.splitext(self.image_path)
        return f"{stem}_recolored.png"

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
        self._original_display_uint8 = _rgb_float_to_uint8(self.preview_frame.rgb_float)

    def _build_default_curve_lines(self) -> list[np.ndarray]:
        """Build the exact baseline curves shown by the dashed reference lines."""
        lightness_line = np.array(CURVE_X_DENSE, copy=True)
        chroma_line = np.clip(self.base_model.c_interp(CURVE_X_DENSE), 0.0, None)
        hue_line = _evaluate_hue_curve(self.base_model.u_interp, self.base_model.v_interp, CURVE_X_DENSE)
        return [lightness_line, chroma_line, hue_line]

    def _sample_default_curve_points(self, curve_index: int, point_count: int | None = None) -> np.ndarray:
        """Sample sparse editor control points from the exact baseline curve."""
        resolved_count = max(CURVE_EDITOR_MIN_CTRL_POINTS, int(point_count or STATE_CURVE_CTRL_POINTS))
        x_values = np.linspace(0.0, 1.0, resolved_count, dtype=np.float64)
        if curve_index == 0:
            y_values = np.array(x_values, copy=True)
        elif curve_index == 1:
            y_values = np.clip(self.base_model.c_interp(x_values), 0.0, None)
        else:
            y_values = _evaluate_hue_curve(self.base_model.u_interp, self.base_model.v_interp, x_values)
        return np.column_stack([x_values, y_values])

    def _prepare_curve_control_points(
        self,
        curve_index: int,
        control_points: np.ndarray,
        *,
        default_points: np.ndarray | None = None,
    ) -> np.ndarray:
        """Normalize one curve's control points and keep endpoints anchored to the full domain."""
        fallback_points = np.asarray(
            default_points if default_points is not None else self._sample_default_curve_points(curve_index),
            dtype=np.float64,
        )
        if curve_index == 0:
            points = prepare_control_points(
                control_points,
                fallback_points[:, 0],
                fallback_points[:, 1],
                clip_min=0.0,
                clip_max=1.0,
            )
        elif curve_index == 1:
            points = prepare_control_points(
                control_points,
                fallback_points[:, 0],
                fallback_points[:, 1],
                clip_min=0.0,
            )
        else:
            points = prepare_control_points(
                control_points,
                fallback_points[:, 0],
                fallback_points[:, 1],
                wrap_degrees=True,
            )

        points = np.array(points, copy=True)
        points[0, 0] = 0.0
        points[-1, 0] = 1.0
        return points

    def _sample_effective_curve_values(
        self,
        curve_index: int,
        x_values: np.ndarray,
        state_curves=None,
    ) -> np.ndarray:
        """Sample one curve from the currently effective state-curve set."""
        resolved_state_curves = state_curves or self._build_state_curves()
        x_values = np.clip(np.asarray(x_values, dtype=np.float64), 0.0, 1.0)
        if curve_index == 0:
            return np.clip(resolved_state_curves.lightness_interp(x_values), 0.0, 1.0)
        if curve_index == 1:
            return np.clip(resolved_state_curves.chroma_interp(x_values), 0.0, None)
        return _evaluate_hue_curve(resolved_state_curves.hue_u_interp, resolved_state_curves.hue_v_interp, x_values)

    def _current_hue_display_range(self) -> tuple[float, float]:
        """Return the currently visible continuous 360-degree hue window."""
        start = float(self.hue_display_start)
        return (start, start + HUE_DISPLAY_WINDOW_SPAN)

    def _map_hue_values_to_display_window(
        self,
        hue_values: np.ndarray,
        *,
        start: float | None = None,
    ) -> np.ndarray:
        """Map canonical [0, 360) hue values into the active shifted display window."""
        resolved_start = float(self.hue_display_start if start is None else start)
        hue_values = np.asarray(hue_values, dtype=np.float64)
        return np.mod(hue_values - resolved_start, HUE_DISPLAY_WINDOW_SPAN) + resolved_start

    def _sync_hue_display_controls(self):
        """Keep the hue-window widgets aligned with the current display start."""
        slider = getattr(self, "hue_display_start_slider", None)
        if slider is not None:
            blocker = QtCore.QSignalBlocker(slider)
            slider.setValue(int(round(self.hue_display_start)))
            del blocker

        value_label = getattr(self, "hue_display_start_value_label", None)
        if value_label is not None:
            start = int(round(self.hue_display_start))
            value_label.setText(f"{start} to {start + int(HUE_DISPLAY_WINDOW_SPAN)} deg")

    def _refresh_hue_plot_display(
        self,
        *,
        state_curves=None,
        hue_line: np.ndarray | None = None,
    ):
        """Refresh only the hue plot after display-window or control-point changes."""
        if not hasattr(self, "hue_plot"):
            return

        resolved_state_curves = state_curves or self._build_state_curves()
        resolved_hue_line = (
            np.asarray(hue_line, dtype=np.float64)
            if hue_line is not None
            else self._sample_effective_curve_values(2, CURVE_X_DENSE, state_curves=resolved_state_curves)
        )
        self.hue_plot.set_y_range(self._current_hue_display_range())
        self.hue_plot.set_background_rgb(self._build_hue_background(resolved_state_curves))
        self.hue_plot.set_default_line(self._map_hue_values_to_display_window(self._default_curve_lines[2]))
        self.hue_plot.set_curve_line(self._map_hue_values_to_display_window(resolved_hue_line))
        self.hue_plot.set_control_points(
            self.ctrl_x[2],
            self._map_hue_values_to_display_window(self.ctrl_y[2]),
        )
        self.hue_plot.set_histogram(None, None)
        self.hue_plot.set_reference_histogram(None, None)
        self._sync_hue_display_controls()

    def _set_hue_display_start(self, start_degrees: int):
        """Shift the visible hue window without modifying the stored curve."""
        resolved_start = int(np.clip(int(start_degrees), HUE_DISPLAY_START_MIN, HUE_DISPLAY_START_MAX))
        if resolved_start == int(round(self.hue_display_start)):
            self._sync_hue_display_controls()
            return
        self.hue_display_start = float(resolved_start)
        self._refresh_hue_plot_display()

    def _build_curve_panel(self, curve_index: int, plot_widget: CurvePlotWidget) -> QtWidgets.QWidget:
        """Wrap one plot with point-count and reset controls."""
        panel = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(6)

        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        points_label = QtWidgets.QLabel("Key Points")
        points_spin_box = QtWidgets.QSpinBox()
        points_spin_box.setRange(CURVE_EDITOR_MIN_CTRL_POINTS, CURVE_EDITOR_MAX_CTRL_POINTS)
        points_spin_box.setValue(int(self.ctrl_x[curve_index].size))
        points_spin_box.setToolTip("Change how many sparse key points are exposed for this curve.")

        reset_button = QtWidgets.QPushButton("Reset to Default")
        reset_button.setToolTip("Use the original baseline curve again, ignoring the current sparse key-point approximation.")

        mode_label = QtWidgets.QLabel()
        mode_label.setStyleSheet("color: #aaaaaa;")

        controls_layout.addWidget(points_label)
        controls_layout.addWidget(points_spin_box)
        controls_layout.addWidget(reset_button)
        controls_layout.addStretch(1)
        controls_layout.addWidget(mode_label)

        panel_layout.addLayout(controls_layout)
        if curve_index == 2:
            hue_window_layout = QtWidgets.QHBoxLayout()
            hue_window_layout.setContentsMargins(0, 0, 0, 0)
            hue_window_layout.setSpacing(8)

            hue_window_label = QtWidgets.QLabel("Hue Window Start")
            hue_window_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            hue_window_slider.setRange(HUE_DISPLAY_START_MIN, HUE_DISPLAY_START_MAX)
            hue_window_slider.setSingleStep(1)
            hue_window_slider.setPageStep(15)
            hue_window_slider.setValue(int(round(self.hue_display_start)))
            hue_window_slider.setToolTip(
                "Shift the visible 360-degree hue window without changing the stored hue curve."
            )

            hue_window_value_label = QtWidgets.QLabel()
            hue_window_value_label.setMinimumWidth(120)

            hue_window_layout.addWidget(hue_window_label)
            hue_window_layout.addWidget(hue_window_slider, 1)
            hue_window_layout.addWidget(hue_window_value_label)
            panel_layout.addLayout(hue_window_layout)

            hue_window_slider.valueChanged.connect(self._set_hue_display_start)
            self.hue_display_start_slider = hue_window_slider
            self.hue_display_start_value_label = hue_window_value_label
            self._sync_hue_display_controls()

        panel_layout.addWidget(plot_widget, 1)

        points_spin_box.valueChanged.connect(
            lambda value, idx=curve_index: self._set_curve_point_count(idx, value)
        )
        reset_button.clicked.connect(
            lambda _checked=False, idx=curve_index: self._reset_curve_to_default(idx)
        )

        self.curve_point_count_spinboxes.append(points_spin_box)
        self.curve_reset_buttons.append(reset_button)
        self.curve_mode_labels.append(mode_label)
        self._sync_curve_editor_controls(curve_index)
        return panel

    def _sync_curve_editor_controls(self, curve_index: int):
        """Keep the point-count widgets and mode labels aligned with the current curve state."""
        if not hasattr(self, "curve_point_count_spinboxes") or curve_index >= len(self.curve_point_count_spinboxes):
            return

        point_count_spin_box = self.curve_point_count_spinboxes[curve_index]
        blocker = QtCore.QSignalBlocker(point_count_spin_box)
        point_count_spin_box.setValue(int(self.ctrl_x[curve_index].size))
        del blocker

        mode_label = self.curve_mode_labels[curve_index]
        if self._curve_override_enabled[curve_index]:
            mode_label.setText(CURVE_EDITOR_MODE_EDITED)
            mode_label.setStyleSheet("color: #d7d7d7;")
        else:
            mode_label.setText(CURVE_EDITOR_MODE_DEFAULT)
            mode_label.setStyleSheet("color: #8fe388;")

    def _set_curve_point_count(self, curve_index: int, point_count: int):
        """Resample the visible key points to a requested sparse count."""
        resolved_count = max(CURVE_EDITOR_MIN_CTRL_POINTS, int(point_count))
        current_count = int(self.ctrl_x[curve_index].size)
        if resolved_count == current_count:
            self._sync_curve_editor_controls(curve_index)
            return

        x_values = np.linspace(0.0, 1.0, resolved_count, dtype=np.float64)
        if self._curve_override_enabled[curve_index]:
            state_curves = self._last_state_curves or self._build_state_curves()
            y_values = self._sample_effective_curve_values(curve_index, x_values, state_curves=state_curves)
            override_enabled = True
        else:
            default_points = self._sample_default_curve_points(curve_index, resolved_count)
            y_values = default_points[:, 1]
            override_enabled = False

        self._apply_control_points(
            curve_index,
            np.column_stack([x_values, y_values]),
            rerender=True,
            override_enabled=override_enabled,
        )

    def _reset_curve_to_default(self, curve_index: int):
        """Revert one curve back to the exact default baseline while keeping sparse handles visible."""
        default_points = self._sample_default_curve_points(curve_index, self.ctrl_x[curve_index].size)
        self._apply_control_points(
            curve_index,
            default_points,
            rerender=True,
            override_enabled=False,
        )

    def _sample_initial_curves(self, initial_curve_overrides: dict):
        sampled_points: list[np.ndarray] = []
        self._curve_override_enabled = []
        for curve_index, override_key in enumerate(CURVE_EDITOR_OVERRIDE_KEYS):
            override_points = initial_curve_overrides.get(override_key)
            if override_points is None:
                sampled_points.append(self._sample_default_curve_points(curve_index, STATE_CURVE_CTRL_POINTS))
                self._curve_override_enabled.append(False)
                continue

            override_points = np.asarray(override_points, dtype=np.float64)
            default_points = self._sample_default_curve_points(curve_index, override_points.shape[0])
            sampled_points.append(
                self._prepare_curve_control_points(curve_index, override_points, default_points=default_points)
            )
            self._curve_override_enabled.append(True)
        return tuple(sampled_points)

    def _initialize_controls(self, initial_curve_overrides: dict):
        lightness_points, chroma_points, hue_points = self._sample_initial_curves(initial_curve_overrides)
        self.ctrl_x = [lightness_points[:, 0], chroma_points[:, 0], hue_points[:, 0]]
        self.ctrl_y = [lightness_points[:, 1], chroma_points[:, 1], hue_points[:, 1]]

        base_chroma_max = max(np.max(self.base_model.key_c), np.max(chroma_points[:, 1]), 1e-3)
        self.chroma_ylim = (0.0, max(0.35, float(base_chroma_max) * 1.25))

    def _current_control_point_payload(self) -> dict:
        payload = {}
        for curve_index, override_key in enumerate(CURVE_EDITOR_OVERRIDE_KEYS):
            if self._curve_override_enabled[curve_index]:
                payload[override_key] = np.column_stack([self.ctrl_x[curve_index], self.ctrl_y[curve_index]])
            else:
                payload[override_key] = None
        return payload

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
        self.setWindowTitle("Texture-Map-Toolbox Qt MVP")
        self.resize(1680, 980)

        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        root_layout = QtWidgets.QVBoxLayout(central_widget)

        action_layout = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton("Save Curves JSON")
        self.export_image_button = QtWidgets.QPushButton("Export Image")
        self.render_button = QtWidgets.QPushButton("Full-Resolution Render")
        self.load_target_picker_button = QtWidgets.QPushButton("Load L/C/H Target")
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("color: #ddd;")
        action_layout.addWidget(self.save_button)
        action_layout.addWidget(self.export_image_button)
        action_layout.addWidget(self.render_button)
        action_layout.addWidget(self.load_target_picker_button)
        action_layout.addWidget(self.status_label, 1)
        root_layout.addLayout(action_layout)

        content_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root_layout.addWidget(content_splitter, 1)

        curves_widget = QtWidgets.QWidget()
        curves_layout = QtWidgets.QVBoxLayout(curves_widget)
        curves_layout.setContentsMargins(0, 0, 0, 0)
        curves_layout.setSpacing(10)
        self.curve_point_count_spinboxes: list[QtWidgets.QSpinBox] = []
        self.curve_reset_buttons: list[QtWidgets.QPushButton] = []
        self.curve_mode_labels: list[QtWidgets.QLabel] = []

        self.lightness_plot = CurvePlotWidget(
            title="Lightness Transfer Lt(y)",
            x_label="Input Lightness L0",
            y_label="Output Lightness L'",
            y_range=(0.0, 1.0),
            fixed_x=self.ctrl_x[0],
            line_pen=pg.mkPen("#f5c842", width=2.5),
            point_brush=pg.mkBrush("#f5c842"),
        )
        self.chroma_plot = CurvePlotWidget(
            title="Chroma State Ct(L')",
            x_label="Output Lightness L'",
            y_label="Output Chroma C'",
            y_range=self.chroma_ylim,
            fixed_x=self.ctrl_x[1],
            line_pen=pg.mkPen("#2db7ff", width=2.5),
            point_brush=pg.mkBrush("#2db7ff"),
        )
        self.hue_plot = CurvePlotWidget(
            title="Hue State ht(L')",
            x_label="Output Lightness L'",
            y_label="Output Hue h' (deg)",
            y_range=self._current_hue_display_range(),
            fixed_x=self.ctrl_x[2],
            line_pen=pg.mkPen("#ff6b57", width=2.5),
            point_brush=pg.mkBrush("#ff6b57"),
        )

        self.lightness_plot.set_default_line(self._default_curve_lines[0])
        self.chroma_plot.set_default_line(self._default_curve_lines[1])
        self.hue_plot.set_default_line(self._map_hue_values_to_display_window(self._default_curve_lines[2]))

        curves_layout.addWidget(self._build_curve_panel(0, self.lightness_plot), 1)
        curves_layout.addWidget(self._build_curve_panel(1, self.chroma_plot), 1)
        curves_layout.addWidget(self._build_curve_panel(2, self.hue_plot), 1)

        images_widget = QtWidgets.QWidget()
        images_layout = QtWidgets.QHBoxLayout(images_widget)

        original_group = QtWidgets.QGroupBox("Original Image")
        original_layout = QtWidgets.QVBoxLayout(original_group)
        self.original_image_label = ImagePreviewLabel("Original")
        original_layout.addWidget(self.original_image_label)

        preview_group = QtWidgets.QGroupBox("Preview Image")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        self.preview_image_label = ImagePreviewLabel("Preview")
        preview_layout.addWidget(self.preview_image_label)

        images_layout.addWidget(original_group, 1)
        images_layout.addWidget(preview_group, 1)

        content_splitter.addWidget(curves_widget)
        content_splitter.addWidget(images_widget)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 2)

        self.original_image_label.set_rgb_uint8(self._original_display_uint8)

        self.lightness_plot.control_item.sigControlPointPositionsChanged.connect(
            lambda positions: self._on_curve_points_changed(0, positions)
        )
        self.chroma_plot.control_item.sigControlPointPositionsChanged.connect(
            lambda positions: self._on_curve_points_changed(1, positions)
        )
        self.hue_plot.control_item.sigControlPointPositionsChanged.connect(
            lambda positions: self._on_curve_points_changed(2, positions)
        )
        self.save_button.clicked.connect(self._save_curves)
        self.export_image_button.clicked.connect(self._export_full_resolution_image)
        self.render_button.clicked.connect(self._render_full_resolution)
        self.load_target_picker_button.clicked.connect(self._open_target_image_picker)

    def _on_curve_points_changed(self, curve_index: int, control_points: np.ndarray):
        """Update one curve from dragged control points and rerender."""
        self._apply_control_points(
            curve_index,
            np.asarray(control_points, dtype=np.float64),
            rerender=True,
            override_enabled=True,
        )

    def _show_error(self, title: str, message: str):
        """Display a blocking Qt error dialog and mirror it in the status bar."""
        self.status_label.setText(message)
        QtWidgets.QMessageBox.critical(self, title, message)

    def _show_image_warnings(self, title: str, warnings: tuple[str, ...]):
        """Show image warnings when this window is configured to surface them."""
        if not getattr(self, "show_warning_dialogs", False) or not warnings:
            return
        QtWidgets.QMessageBox.warning(self, title, "\n\n".join(warnings))

    def _load_target_curve_model(
        self,
        image_path: str,
        *,
        alpha_mask_path: str | None = None,
        mask_seed_points: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
        mask_seed_point: tuple[int, int] | None = None,
        mask_color_tolerance: int = DEFAULT_SEED_MASK_COLOR_TOLERANCE,
        mask_region_offset: int = 0,
        show_warnings: bool = True,
    ) -> tuple[LoadedImageData, OklchCurveModel, np.ndarray]:
        """Load a target image and build its Oklch curve model."""
        loaded_image = load_image_data(
            image_path,
            alpha_mask_path=alpha_mask_path,
            mask_seed_points=mask_seed_points,
            mask_seed_point=mask_seed_point,
            mask_color_tolerance=mask_color_tolerance,
            mask_region_offset=mask_region_offset,
        )
        if show_warnings:
            self._show_image_warnings("Target Image Warning", loaded_image.image_warnings)
        target_model, _ = build_oklch_curve_model(loaded_image.oklch_float, loaded_image.valid_mask)
        target_lightness_samples = loaded_image.oklch_float[loaded_image.valid_mask, 0]
        return loaded_image, target_model, target_lightness_samples

    def _apply_control_points(
        self,
        curve_index: int,
        control_points: np.ndarray,
        *,
        rerender: bool = True,
        override_enabled: bool | None = True,
    ):
        """Replace one curve's control points, optionally toggling default-curve mode."""
        control_points = np.asarray(control_points, dtype=np.float64)
        default_points = self._sample_default_curve_points(curve_index, control_points.shape[0])
        normalized_points = self._prepare_curve_control_points(
            curve_index,
            control_points,
            default_points=default_points,
        )
        self.ctrl_x[curve_index] = np.asarray(normalized_points[:, 0], dtype=np.float64)
        self.ctrl_y[curve_index] = np.asarray(normalized_points[:, 1], dtype=np.float64)
        if override_enabled is not None:
            self._curve_override_enabled[curve_index] = bool(override_enabled)
        if curve_index == 1:
            chroma_max = max(float(np.max(self.ctrl_y[1])), float(np.max(self.base_model.key_c)), 1e-3)
            self.chroma_ylim = (0.0, max(0.35, chroma_max * 1.25))
            if hasattr(self, "chroma_plot"):
                self.chroma_plot.set_y_range(self.chroma_ylim)
        self._sync_curve_editor_controls(curve_index)
        if rerender:
            self._render_preview()

    def apply_target_image_selection(
        self,
        image_path: str,
        *,
        alpha_mask_path: str | None = None,
        mask_seed_points: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
        mask_seed_point: tuple[int, int] | None = None,
        mask_color_tolerance: int = DEFAULT_SEED_MASK_COLOR_TOLERANCE,
        mask_region_offset: int = 0,
        apply_lightness: bool,
        apply_chroma: bool,
        apply_hue: bool,
        show_warnings: bool = True,
    ):
        """Load one target image and apply it to the selected L/C/H curves."""
        if not any((apply_lightness, apply_chroma, apply_hue)):
            raise ValueError("at least one target curve must be selected")

        loaded_image, target_model, target_lightness_samples = self._load_target_curve_model(
            image_path,
            alpha_mask_path=alpha_mask_path,
            mask_seed_points=mask_seed_points,
            mask_seed_point=mask_seed_point,
            mask_color_tolerance=mask_color_tolerance,
            mask_region_offset=mask_region_offset,
            show_warnings=show_warnings,
        )
        self._last_target_image_dialog_path = loaded_image.image_path
        self._last_target_mask_dialog_path = loaded_image.alpha_mask_path
        if loaded_image.alpha_source == "external-mask":
            self._last_target_mask_mode = "external"
        elif loaded_image.alpha_source == "interactive-seed":
            self._last_target_mask_mode = "interactive-seed"
        else:
            self._last_target_mask_mode = "image-alpha"

        applied_labels: list[str] = []
        if apply_lightness:
            lightness_point_count = int(self.ctrl_x[0].size)
            fitted_lightness_points = fit_monotonic_lightness_transfer_curve(
                self.source_lightness_samples,
                target_lightness_samples,
                quantile_count=lightness_point_count,
            )
            lightness_x = np.linspace(0.0, 1.0, lightness_point_count, dtype=np.float64)
            lightness_interp = PchipInterpolator(
                fitted_lightness_points[:, 0],
                fitted_lightness_points[:, 1],
                extrapolate=True,
            )
            lightness_points = np.column_stack([
                lightness_x,
                np.clip(lightness_interp(lightness_x), 0.0, 1.0),
            ])
            self.target_curve_image_paths["lightness"] = loaded_image.image_path
            self.target_curve_mask_paths["lightness"] = loaded_image.alpha_mask_path
            self._lightness_reference_histogram = self._build_lightness_histogram(target_lightness_samples)
            self._apply_control_points(0, lightness_points, rerender=False, override_enabled=True)
            applied_labels.append("L")

        if apply_chroma:
            chroma_x = np.linspace(0.0, 1.0, int(self.ctrl_x[1].size), dtype=np.float64)
            chroma_points = np.column_stack([
                chroma_x,
                np.clip(target_model.c_interp(chroma_x), 0.0, None),
            ])
            self.target_curve_image_paths["chroma"] = loaded_image.image_path
            self.target_curve_mask_paths["chroma"] = loaded_image.alpha_mask_path
            self._apply_control_points(1, chroma_points, rerender=False, override_enabled=True)
            applied_labels.append("C")

        if apply_hue:
            hue_x = np.linspace(0.0, 1.0, int(self.ctrl_x[2].size), dtype=np.float64)
            hue_points = np.column_stack([
                hue_x,
                _evaluate_hue_curve(target_model.u_interp, target_model.v_interp, hue_x),
            ])
            self.target_curve_image_paths["hue"] = loaded_image.image_path
            self.target_curve_mask_paths["hue"] = loaded_image.alpha_mask_path
            self._apply_control_points(2, hue_points, rerender=False, override_enabled=True)
            applied_labels.append("H")

        self._render_preview()
        self.status_label.setText(
            "Loaded target image for {}: {}".format("/".join(applied_labels), loaded_image.image_path)
        )

    def apply_lightness_target_image(self, image_path: str, *, alpha_mask_path: str | None = None):
        """Load a target image for Lt(y) and fit a monotonic lightness transfer curve."""
        self.apply_target_image_selection(
            image_path,
            alpha_mask_path=alpha_mask_path,
            apply_lightness=True,
            apply_chroma=False,
            apply_hue=False,
        )

    def apply_chroma_target_image(self, image_path: str, *, alpha_mask_path: str | None = None):
        """Load a target image and copy its chroma curve into Ct(L')."""
        self.apply_target_image_selection(
            image_path,
            alpha_mask_path=alpha_mask_path,
            apply_lightness=False,
            apply_chroma=True,
            apply_hue=False,
        )

    def apply_hue_target_image(self, image_path: str, *, alpha_mask_path: str | None = None):
        """Load a target image and copy its hue curve into ht(L')."""
        self.apply_target_image_selection(
            image_path,
            alpha_mask_path=alpha_mask_path,
            apply_lightness=False,
            apply_chroma=False,
            apply_hue=True,
        )

    def _open_target_image_picker(self):
        """Choose one target image and optionally apply it to L/C/H together."""
        dialog = QtTargetImagePickerDialog(
            self,
            initial_image_path=self._last_target_image_dialog_path,
            initial_mask_path=self._last_target_mask_dialog_path,
            initial_mask_mode=self._last_target_mask_mode,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        selected_path = dialog.selected_image_path()
        selected_mask_path = dialog.selected_mask_path()
        mask_seed_points = dialog.selected_mask_seed_points()
        apply_lightness, apply_chroma, apply_hue = dialog.selected_curve_flags()
        if selected_path is None:
            return
        try:
            self.apply_target_image_selection(
                selected_path,
                alpha_mask_path=selected_mask_path,
                mask_seed_points=mask_seed_points,
                mask_color_tolerance=dialog.selected_mask_color_tolerance(),
                mask_region_offset=dialog.selected_mask_region_offset(),
                apply_lightness=apply_lightness,
                apply_chroma=apply_chroma,
                apply_hue=apply_hue,
                show_warnings=True,
            )
        except Exception as exc:  # noqa: BLE001
            self._show_error("Load Target Image Failed", str(exc))

    def _build_lightness_background(self) -> np.ndarray:
        """Build the grayscale lightness-panel background."""
        lightness_values = np.linspace(0.0, 1.0, CURVE_BACKGROUND_HEIGHT, dtype=np.float64)[:, None, None]
        gray_uint8 = np.clip(np.round(lightness_values * 255.0), 0.0, 255.0).astype(np.uint8)
        return np.repeat(np.repeat(gray_uint8, CURVE_BACKGROUND_WIDTH, axis=1), 3, axis=2)

    def _build_chroma_background(self, state_curves) -> np.ndarray:
        """Build the chroma-panel background using current hue along x."""
        lightness_x = np.linspace(0.0, 1.0, CURVE_BACKGROUND_WIDTH, dtype=np.float64)
        chroma_y = np.linspace(0.0, self.chroma_ylim[1], CURVE_BACKGROUND_HEIGHT, dtype=np.float64)[:, None]
        hue_x = _evaluate_hue_curve(state_curves.hue_u_interp, state_curves.hue_v_interp, lightness_x)

        lightness_grid = np.broadcast_to(lightness_x[None, :], (CURVE_BACKGROUND_HEIGHT, CURVE_BACKGROUND_WIDTH))
        chroma_grid = np.broadcast_to(chroma_y, (CURVE_BACKGROUND_HEIGHT, CURVE_BACKGROUND_WIDTH))
        hue_grid = np.broadcast_to(hue_x[None, :], (CURVE_BACKGROUND_HEIGHT, CURVE_BACKGROUND_WIDTH))
        oklch_grid = np.stack([lightness_grid, chroma_grid, hue_grid], axis=-1)
        _, rgb_float, _ = compress_oklch_chroma_to_srgb(oklch_grid)
        return _rgb_float_to_uint8(rgb_float)

    def _build_hue_background(self, state_curves) -> np.ndarray:
        """Build the hue-panel background using current chroma along x."""
        lightness_x = np.linspace(0.0, 1.0, CURVE_BACKGROUND_WIDTH, dtype=np.float64)
        hue_y = np.linspace(*self._current_hue_display_range(), CURVE_BACKGROUND_HEIGHT, dtype=np.float64)[:, None]
        chroma_x = np.clip(state_curves.chroma_interp(lightness_x), 0.0, None)

        lightness_grid = np.broadcast_to(lightness_x[None, :], (CURVE_BACKGROUND_HEIGHT, CURVE_BACKGROUND_WIDTH))
        chroma_grid = np.broadcast_to(chroma_x[None, :], (CURVE_BACKGROUND_HEIGHT, CURVE_BACKGROUND_WIDTH))
        hue_grid = np.mod(
            np.broadcast_to(hue_y, (CURVE_BACKGROUND_HEIGHT, CURVE_BACKGROUND_WIDTH)),
            HUE_DISPLAY_WINDOW_SPAN,
        )
        oklch_grid = np.stack([lightness_grid, chroma_grid, hue_grid], axis=-1)
        _, rgb_float, _ = compress_oklch_chroma_to_srgb(oklch_grid)
        return _rgb_float_to_uint8(rgb_float)

    def _build_lightness_histogram(self, lightness_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build a compact histogram overlay for the lightness panel."""
        histogram, bin_edges = np.histogram(lightness_values, bins=LIGHTNESS_HISTOGRAM_BINS, range=(0.0, 1.0))
        histogram = histogram.astype(np.float64)
        if np.max(histogram) > 0.0:
            histogram = histogram / np.max(histogram) * 0.28
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return bin_centers, histogram

    def _render_preview(self):
        """Recompute the preview image and all three curve panels."""
        start = time.perf_counter()
        state_curves = self._build_state_curves()
        lightness_line, chroma_line, hue_line = self._sample_curve_lines(state_curves)

        preview_y_eval = apply_precurve_dither(self.y_small, self.mask_small, self.dither_strength)
        if self.dither_strength > 0.0:
            preview_y_index = compute_luma_lut_indices(preview_y_eval, PREVIEW_LUT_SIZE)
        else:
            preview_y_index = self.y_small_index

        preview_lut_uint8, gamut_pixels = self._build_preview_lut(state_curves, preview_y_index)
        lut_done = time.perf_counter()
        apply_luma_preview_lut(preview_y_index, self._preview_output_mask, preview_lut_uint8, out_buf=self._preview_buf)
        recolor_done = time.perf_counter()

        preview_lightness = np.clip(state_curves.lightness_interp(preview_y_eval[self.mask_small]), 0.0, 1.0)
        hist_x, hist_y = self._build_lightness_histogram(preview_lightness)

        self.lightness_plot.set_background_rgb(self._build_lightness_background())
        self.lightness_plot.set_curve_line(lightness_line)
        self.lightness_plot.set_control_points(self.ctrl_x[0], self.ctrl_y[0])
        self.lightness_plot.set_histogram(hist_x, hist_y)
        if self._lightness_reference_histogram is None:
            self.lightness_plot.set_reference_histogram(None, None)
        else:
            self.lightness_plot.set_reference_histogram(*self._lightness_reference_histogram)

        self.chroma_plot.set_background_rgb(self._build_chroma_background(state_curves))
        self.chroma_plot.set_curve_line(chroma_line)
        self.chroma_plot.set_control_points(self.ctrl_x[1], self.ctrl_y[1])
        self.chroma_plot.set_histogram(None, None)
        self.chroma_plot.set_reference_histogram(None, None)

        self._refresh_hue_plot_display(state_curves=state_curves, hue_line=hue_line)

        self.preview_image_label.set_rgb_uint8(self._preview_buf)
        draw_done = time.perf_counter()

        self._last_state_curves = state_curves
        self.status_label.setText(
            "state+lut={:.1f} ms   recolor={:.1f} ms   ui={:.1f} ms   gamut={}"
            .format(
                1000.0 * (lut_done - start),
                1000.0 * (recolor_done - lut_done),
                1000.0 * (draw_done - recolor_done),
                gamut_pixels,
            )
        )

    def _save_curves(self):
        """Export the current Lt/Ct/ht control points to JSON."""
        payload = {}
        for curve_index, save_key in enumerate(CURVE_EDITOR_SAVE_KEYS):
            if not self._curve_override_enabled[curve_index]:
                continue
            payload[save_key] = np.column_stack([self.ctrl_x[curve_index], self.ctrl_y[curve_index]]).tolist()
        with open(self.curve_output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        self.status_label.setText(f"Saved curves: {self.curve_output_path}")

    def _compute_full_resolution_render(self):
        """Run the shared full-resolution reconstruction used by render and export."""
        state_curves = self._last_state_curves or self._build_state_curves()
        recolored_rgb_float, _, y_eval, gamut_pixels = reconstruct_from_state_curves(
            self.oklch_float,
            self._output_valid_mask,
            state_curves,
            dither_strength=self.dither_strength,
        )
        recolored_rgb_int, psnr, delta_e_image, delta_e_stats = evaluate_reconstruction(
            self.rgb_float,
            recolored_rgb_float,
            self.valid_mask,
        )
        return recolored_rgb_int, y_eval, gamut_pixels, psnr, delta_e_image, delta_e_stats

    def _select_output_image_path(self) -> str | None:
        selected_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export recolored image",
            self.output_image_path,
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;BMP Files (*.bmp);;TIFF Files (*.tif *.tiff);;WebP Files (*.webp);;All Files (*)",
        )
        if not selected_path:
            return None
        self.output_image_path = selected_path
        return selected_path

    def _export_full_resolution_image(self):
        """Render the current state curves at full resolution and save the output image."""
        output_path = self._select_output_image_path()
        if output_path is None:
            self.status_label.setText("Image export canceled")
            return

        try:
            recolored_rgb_int, _, gamut_pixels, psnr, _, delta_e_stats = self._compute_full_resolution_render()
            save_luma_output_image(recolored_rgb_int, output_path)
        except Exception as exc:  # noqa: BLE001
            self._show_error("Export Image Failed", str(exc))
            return

        self.status_label.setText(
            "Exported image: {}  gamut={}  PSNR={:.2f} dB  DeltaE mean={:.2f}"
            .format(output_path, gamut_pixels, psnr, delta_e_stats["mean"])
        )

    def _render_full_resolution(self):
        """Run the original full-resolution reconstruction and display comparison plots."""
        recolored_rgb_int, y_eval, gamut_pixels, psnr, delta_e_image, delta_e_stats = self._compute_full_resolution_render()
        plot_comparison(self.rgb_float, y_eval, recolored_rgb_int, self.valid_mask, psnr, delta_e_image)
        show_figures(block=False)
        self.status_label.setText(
            "Full render complete: gamut={}  PSNR={:.2f} dB  DeltaE mean={:.2f}"
            .format(gamut_pixels, psnr, delta_e_stats["mean"])
        )


class QtEditorLauncherWindow(QtWidgets.QWidget):
    """Small launcher window that lets users choose files before opening the Qt editor."""

    def __init__(
        self,
        *,
        image_path: str | None = None,
        alpha_mask_path: str | None = None,
        curve_path: str | None = None,
        curve_output_path: str | None = None,
        dither_strength: float = DITHER_STRENGTH,
    ):
        super().__init__()
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.dither_strength = float(dither_strength)
        self._mask_seed_points: list[tuple[int, int]] = []
        self._opened_editors: list[QtOklchCurveEditorWindow] = []
        self._build_ui()
        self.image_path_edit.setText(image_path or "")
        self.alpha_mask_path_edit.setText(alpha_mask_path or "")
        self.curve_path_edit.setText(curve_path or "")
        self.curve_output_path_edit.setText(curve_output_path or "")
        self._set_initial_mask_mode("external" if alpha_mask_path else "interactive-seed")
        self._refresh_previews()

    def _build_ui(self):
        self.setWindowTitle("Texture-Map-Toolbox Launcher")
        self.resize(1200, 860)

        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(14)

        title_label = QtWidgets.QLabel("Open Oklch Curve Editor")
        title_font = title_label.font()
        title_font.setPointSize(title_font.pointSize() + 5)
        title_font.setBold(True)
        title_label.setFont(title_font)

        subtitle_label = QtWidgets.QLabel(
            "Select an input image and optional files here, then open the Qt editor without touching CLI arguments."
        )
        subtitle_label.setWordWrap(True)
        subtitle_label.setStyleSheet("color: #cccccc;")

        root_layout.addWidget(title_label)
        root_layout.addWidget(subtitle_label)

        content_layout = QtWidgets.QHBoxLayout()
        content_layout.setSpacing(18)
        root_layout.addLayout(content_layout, 1)

        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(12)
        content_layout.addWidget(controls_widget, 1)

        form_layout = QtWidgets.QGridLayout()
        form_layout.setHorizontalSpacing(10)
        form_layout.setVerticalSpacing(10)
        controls_layout.addLayout(form_layout)

        self.image_path_edit = QtWidgets.QLineEdit()
        self.alpha_mask_path_edit = QtWidgets.QLineEdit()
        self.curve_path_edit = QtWidgets.QLineEdit()
        self.curve_output_path_edit = QtWidgets.QLineEdit()

        self.image_path_edit.setPlaceholderText("Required input image")
        self.alpha_mask_path_edit.setPlaceholderText("Optional alpha mask image")
        self.curve_path_edit.setPlaceholderText("Optional initial curves JSON")
        self.curve_output_path_edit.setPlaceholderText("Optional curve export JSON path")

        _, self.use_sample_button = self._add_path_row(
            form_layout,
            row=0,
            label_text="Input Image",
            line_edit=self.image_path_edit,
            browse_handler=self._browse_input_image,
            secondary_text="Use Sample",
            secondary_handler=self._use_bundled_sample,
        )

        mask_mode_group = QtWidgets.QGroupBox("Mask Source")
        mask_mode_layout = QtWidgets.QVBoxLayout(mask_mode_group)
        mask_mode_layout.setSpacing(8)
        self.use_image_alpha_radio = QtWidgets.QRadioButton("Use image alpha / no extra mask")
        self.load_mask_radio = QtWidgets.QRadioButton("Load mask file")
        self.pick_region_radio = QtWidgets.QRadioButton("Pick connected region from image")
        mask_mode_layout.addWidget(self.use_image_alpha_radio)
        mask_mode_layout.addWidget(self.load_mask_radio)
        mask_mode_layout.addWidget(self.pick_region_radio)
        controls_layout.addWidget(mask_mode_group)

        mask_form_layout = QtWidgets.QGridLayout()
        mask_form_layout.setHorizontalSpacing(10)
        mask_form_layout.setVerticalSpacing(10)
        controls_layout.addLayout(mask_form_layout)
        self.alpha_mask_browse_button, self.alpha_mask_clear_button = self._add_path_row(
            mask_form_layout,
            row=1,
            label_text="Alpha Mask",
            line_edit=self.alpha_mask_path_edit,
            browse_handler=self._browse_alpha_mask,
            secondary_text="Clear",
            secondary_handler=self._clear_alpha_mask,
        )

        self.seed_mask_controls = InteractiveSeedMaskControls()
        controls_layout.addWidget(self.seed_mask_controls)

        file_form_layout = QtWidgets.QGridLayout()
        file_form_layout.setHorizontalSpacing(10)
        file_form_layout.setVerticalSpacing(10)
        controls_layout.addLayout(file_form_layout)
        self._add_path_row(
            file_form_layout,
            row=0,
            label_text="Initial Curves",
            line_edit=self.curve_path_edit,
            browse_handler=self._browse_curve_json,
            secondary_text="Clear",
            secondary_handler=lambda: self.curve_path_edit.clear(),
        )
        self._add_path_row(
            file_form_layout,
            row=1,
            label_text="Curve Output",
            line_edit=self.curve_output_path_edit,
            browse_handler=self._browse_curve_output_path,
            secondary_text="Clear",
            secondary_handler=lambda: self.curve_output_path_edit.clear(),
        )

        controls_layout.addStretch(1)

        previews_widget = QtWidgets.QWidget()
        previews_layout = QtWidgets.QHBoxLayout(previews_widget)
        previews_layout.setContentsMargins(0, 0, 0, 0)
        previews_layout.setSpacing(12)
        content_layout.addWidget(previews_widget, 1)

        input_preview_group = QtWidgets.QGroupBox("Input Image Preview")
        input_preview_layout = QtWidgets.QVBoxLayout(input_preview_group)
        self.input_image_preview_label = ImagePreviewLabel("Input Image Preview")
        self.input_image_preview_label.setMinimumSize(320, 320)
        self.input_image_info_label = QtWidgets.QLabel("Choose an input image to preview it.")
        self.input_image_info_label.setWordWrap(True)
        input_preview_layout.addWidget(self.input_image_preview_label, 1)
        input_preview_layout.addWidget(self.input_image_info_label)
        previews_layout.addWidget(input_preview_group, 1)

        mask_preview_group = QtWidgets.QGroupBox("Mask Preview")
        mask_preview_layout = QtWidgets.QVBoxLayout(mask_preview_group)
        self.mask_preview_label = ImagePreviewLabel("Mask Preview")
        self.mask_preview_label.setMinimumSize(320, 320)
        self.mask_preview_info_label = QtWidgets.QLabel("Choose a mask source to preview it.")
        self.mask_preview_info_label.setWordWrap(True)
        mask_preview_layout.addWidget(self.mask_preview_label, 1)
        mask_preview_layout.addWidget(self.mask_preview_info_label)
        previews_layout.addWidget(mask_preview_group, 1)

        action_layout = QtWidgets.QHBoxLayout()
        action_layout.addStretch(1)
        self.open_editor_button = QtWidgets.QPushButton("Open Qt Editor")
        self.close_button = QtWidgets.QPushButton("Close")
        action_layout.addWidget(self.open_editor_button)
        action_layout.addWidget(self.close_button)
        root_layout.addLayout(action_layout)

        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("color: #cccccc;")
        root_layout.addWidget(self.status_label)

        self.open_editor_button.clicked.connect(self.launch_selected_editor)
        self.close_button.clicked.connect(self.close)
        self.image_path_edit.textChanged.connect(self._on_image_path_changed)
        self.alpha_mask_path_edit.textChanged.connect(self._refresh_mask_preview)
        self.use_image_alpha_radio.toggled.connect(self._on_mask_mode_changed)
        self.load_mask_radio.toggled.connect(self._on_mask_mode_changed)
        self.pick_region_radio.toggled.connect(self._on_mask_mode_changed)
        self.input_image_preview_label.image_point_clicked.connect(self._handle_image_preview_click)
        self.seed_mask_controls.values_changed.connect(self._refresh_previews)
        self.seed_mask_controls.clear_requested.connect(self._clear_seed_points)

    def _add_path_row(
        self,
        form_layout: QtWidgets.QGridLayout,
        *,
        row: int,
        label_text: str,
        line_edit: QtWidgets.QLineEdit,
        browse_handler,
        secondary_text: str,
        secondary_handler,
    ):
        label = QtWidgets.QLabel(label_text)
        browse_button = QtWidgets.QPushButton("Browse")
        secondary_button = QtWidgets.QPushButton(secondary_text)
        browse_button.clicked.connect(browse_handler)
        secondary_button.clicked.connect(secondary_handler)

        form_layout.addWidget(label, row, 0)
        form_layout.addWidget(line_edit, row, 1)
        form_layout.addWidget(browse_button, row, 2)
        form_layout.addWidget(secondary_button, row, 3)
        return browse_button, secondary_button

    def _set_initial_mask_mode(self, initial_mask_mode: str):
        if initial_mask_mode == "external":
            self.load_mask_radio.setChecked(True)
        elif initial_mask_mode == "interactive-seed":
            self.pick_region_radio.setChecked(True)
        else:
            self.use_image_alpha_radio.setChecked(True)

    def _mask_mode(self) -> str:
        if self.load_mask_radio.isChecked():
            return "external"
        if self.pick_region_radio.isChecked():
            return "interactive-seed"
        return "image-alpha"

    def _sync_mask_controls(self):
        controls_enabled = self.load_mask_radio.isChecked()
        self.alpha_mask_path_edit.setEnabled(controls_enabled)
        self.alpha_mask_browse_button.setEnabled(controls_enabled)
        self.alpha_mask_clear_button.setEnabled(controls_enabled)
        interactive_seed_enabled = self._mask_mode() == "interactive-seed"
        self.input_image_preview_label.set_click_enabled(interactive_seed_enabled)
        self.seed_mask_controls.setEnabled(interactive_seed_enabled)
        self.seed_mask_controls.set_seed_points(self._mask_seed_points)

    def _on_image_path_changed(self):
        self._mask_seed_points = []
        self._refresh_previews()

    def _on_mask_mode_changed(self):
        self._sync_mask_controls()
        self._refresh_previews()

    def _handle_image_preview_click(self, row: int, column: int):
        if self._mask_mode() != "interactive-seed":
            return
        self._mask_seed_points = _toggle_seed_point(self._mask_seed_points, (row, column))
        self._refresh_previews()

    def _clear_seed_points(self):
        self._mask_seed_points = []
        self._refresh_previews()

    def _clear_alpha_mask(self):
        self.alpha_mask_path_edit.clear()
        self.pick_region_radio.setChecked(True)

    def _refresh_previews(self):
        self._refresh_image_preview()
        self._refresh_mask_preview()

    def _refresh_image_preview(self):
        image_path = self._line_edit_text_or_none(self.image_path_edit)
        if image_path is None:
            _set_preview_panel_state(
                self.input_image_preview_label,
                self.input_image_info_label,
                preview_rgb=None,
                placeholder_text="Input Image Preview",
                info_text="Choose an input image to preview it.",
            )
            return
        try:
            preview_rgb, image_shape = _load_image_preview_uint8(image_path)
        except Exception as exc:  # noqa: BLE001
            _set_preview_panel_state(
                self.input_image_preview_label,
                self.input_image_info_label,
                preview_rgb=None,
                placeholder_text="Input Image Preview",
                info_text=f"Failed to preview input image: {exc}",
                info_style="color: #ff9b9b;",
            )
            return
        if self._mask_mode() == "interactive-seed":
            summary = _format_interactive_seed_summary(
                self._mask_seed_points,
                color_tolerance=self.seed_mask_controls.color_tolerance(),
                region_offset=self.seed_mask_controls.region_offset(),
            )
        else:
            summary = "Previewing the selected input image."
        _set_preview_panel_state(
            self.input_image_preview_label,
            self.input_image_info_label,
            preview_rgb=preview_rgb,
            placeholder_text="Input Image Preview",
            info_text=_format_image_preview_info(image_shape, summary=summary),
            source_image_shape=image_shape,
            marker_points=self._mask_seed_points if self._mask_mode() == "interactive-seed" else (),
        )

    def _refresh_mask_preview(self):
        image_path = self._line_edit_text_or_none(self.image_path_edit)
        expected_shape = _load_optional_image_shape(image_path)
        mask_mode = self._mask_mode()
        if mask_mode == "external":
            mask_path = self._line_edit_text_or_none(self.alpha_mask_path_edit)
            if mask_path is None:
                _set_preview_panel_state(
                    self.mask_preview_label,
                    self.mask_preview_info_label,
                    preview_rgb=None,
                    placeholder_text="Mask Preview",
                    info_text="Choose an external mask file to preview it.",
                )
                return
            try:
                preview_rgb, mask_shape, valid_pixels = _load_mask_preview_uint8(mask_path)
                info_text, info_style = _format_mask_preview_info(
                    mask_shape,
                    valid_pixels,
                    summary="Previewing the selected external mask.",
                    expected_shape=expected_shape,
                )
            except Exception as exc:  # noqa: BLE001
                _set_preview_panel_state(
                    self.mask_preview_label,
                    self.mask_preview_info_label,
                    preview_rgb=None,
                    placeholder_text="Mask Preview",
                    info_text=f"Failed to preview mask: {exc}",
                    info_style="color: #ff9b9b;",
                )
                return
        elif mask_mode == "interactive-seed":
            if image_path is None:
                _set_preview_panel_state(
                    self.mask_preview_label,
                    self.mask_preview_info_label,
                    preview_rgb=None,
                    placeholder_text="Mask Preview",
                    info_text="Choose an input image, then click one or more pixels in the image preview.",
                )
                return
            if not self._mask_seed_points:
                _set_preview_panel_state(
                    self.mask_preview_label,
                    self.mask_preview_info_label,
                    preview_rgb=None,
                    placeholder_text="Mask Preview",
                    info_text="Click one or more pixels in the input image preview to build a connected-region mask.",
                )
                return
            try:
                preview_rgb, mask_shape, valid_pixels, summary = _load_seeded_mask_preview_uint8(
                    image_path,
                    self._mask_seed_points,
                    color_tolerance=self.seed_mask_controls.color_tolerance(),
                    region_offset=self.seed_mask_controls.region_offset(),
                )
                info_text, info_style = _format_mask_preview_info(
                    mask_shape,
                    valid_pixels,
                    summary=summary,
                    expected_shape=expected_shape,
                )
            except Exception as exc:  # noqa: BLE001
                _set_preview_panel_state(
                    self.mask_preview_label,
                    self.mask_preview_info_label,
                    preview_rgb=None,
                    placeholder_text="Mask Preview",
                    info_text=f"Connected-region preview failed: {exc}",
                    info_style="color: #ff9b9b;",
                )
                return
        else:
            if image_path is None:
                _set_preview_panel_state(
                    self.mask_preview_label,
                    self.mask_preview_info_label,
                    preview_rgb=None,
                    placeholder_text="Mask Preview",
                    info_text="Choose an input image to preview its embedded or implicit mask.",
                )
                return
            try:
                preview_rgb, mask_shape, valid_pixels, summary = _load_embedded_or_implicit_mask_preview_uint8(image_path)
                info_text, info_style = _format_mask_preview_info(
                    mask_shape,
                    valid_pixels,
                    summary=summary,
                    expected_shape=expected_shape,
                )
            except Exception as exc:  # noqa: BLE001
                _set_preview_panel_state(
                    self.mask_preview_label,
                    self.mask_preview_info_label,
                    preview_rgb=None,
                    placeholder_text="Mask Preview",
                    info_text=f"Failed to preview mask source: {exc}",
                    info_style="color: #ff9b9b;",
                )
                return

        _set_preview_panel_state(
            self.mask_preview_label,
            self.mask_preview_info_label,
            preview_rgb=preview_rgb,
            placeholder_text="Mask Preview",
            info_text=info_text,
            info_style=info_style,
        )

    def _browse_existing_file(self, title: str, filter_text: str) -> str | None:
        selected_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, os.getcwd(), filter_text)
        if not selected_path:
            return None
        return selected_path

    def _browse_input_image(self):
        selected_path = self._browse_existing_file(
            "Select input image",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if selected_path is not None:
            self.image_path_edit.setText(selected_path)

    def _browse_alpha_mask(self):
        selected_path = self._browse_existing_file(
            "Select alpha mask image",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if selected_path is not None:
            self.load_mask_radio.setChecked(True)
            self.alpha_mask_path_edit.setText(selected_path)

    def _browse_curve_json(self):
        selected_path = self._browse_existing_file(
            "Select curves JSON",
            "JSON Files (*.json);;All Files (*)",
        )
        if selected_path is not None:
            self.curve_path_edit.setText(selected_path)

    def _browse_curve_output_path(self):
        selected_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select curve export path",
            os.getcwd(),
            "JSON Files (*.json);;All Files (*)",
        )
        if selected_path:
            self.curve_output_path_edit.setText(selected_path)

    def _use_bundled_sample(self):
        try:
            self.image_path_edit.setText(resolve_input_image_path(None))
            self.status_label.setText("Selected bundled sample image.")
        except Exception as exc:  # noqa: BLE001
            self._show_error("Sample Image Unavailable", str(exc))

    def _show_error(self, title: str, message: str):
        self.status_label.setText(message)
        QtWidgets.QMessageBox.critical(self, title, message)

    def _line_edit_text_or_none(self, line_edit: QtWidgets.QLineEdit) -> str | None:
        value = line_edit.text().strip()
        return value or None

    def launch_selected_editor(self, *, show_window: bool = True) -> QtOklchCurveEditorWindow | None:
        image_path = self._line_edit_text_or_none(self.image_path_edit)
        if image_path is None:
            self._show_error("Input Image Required", "Please select an input image before opening the editor.")
            return None

        mask_mode = self._mask_mode()
        alpha_mask_path = None
        mask_seed_points = None
        if mask_mode == "external":
            alpha_mask_path = self._line_edit_text_or_none(self.alpha_mask_path_edit)
            if alpha_mask_path is None:
                self._show_error("Alpha Mask Required", "Please choose an alpha mask file first.")
                return None
            image_shape = _load_optional_image_shape(image_path)
            mask_shape = _load_optional_image_shape(alpha_mask_path)
            if image_shape is not None and mask_shape is not None and tuple(image_shape) != tuple(mask_shape):
                self._show_error("Alpha Mask Size Mismatch", "Alpha mask size must match the selected input image.")
                return None
        elif mask_mode == "interactive-seed":
            if not self._mask_seed_points:
                self._show_error("Seed Pixel Required", "Please click one or more pixels in the input image preview first.")
                return None
            mask_seed_points = tuple(self._mask_seed_points)

        try:
            editor_window = build_qt_editor(
                image_path,
                alpha_mask_path=alpha_mask_path,
                mask_seed_points=mask_seed_points,
                mask_color_tolerance=self.seed_mask_controls.color_tolerance(),
                mask_region_offset=self.seed_mask_controls.region_offset(),
                curve_path=self._line_edit_text_or_none(self.curve_path_edit),
                curve_output_path=self._line_edit_text_or_none(self.curve_output_path_edit),
                dither_strength=self.dither_strength,
            )
        except Exception as exc:  # noqa: BLE001
            self._show_error("Open Editor Failed", str(exc))
            return None

        editor_window.show_warning_dialogs = show_window
        if show_window:
            editor_window.show()
            editor_window._show_image_warnings("Input Image Warning", getattr(editor_window, "image_warnings", ()))
        self._opened_editors.append(editor_window)
        editor_window.destroyed.connect(
            lambda *_args: self._opened_editors.remove(editor_window)
            if editor_window in self._opened_editors
            else None
        )
        self.status_label.setText(f"Opened editor for: {editor_window.image_path}")
        if show_window:
            self.close()
        return editor_window


def build_qt_editor(
    image_path: str | None,
    *,
    alpha_mask_path: str | None = None,
    mask_seed_points: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    mask_seed_point: tuple[int, int] | None = None,
    mask_color_tolerance: int = DEFAULT_SEED_MASK_COLOR_TOLERANCE,
    mask_region_offset: int = 0,
    curve_path: str | None = None,
    curve_output_path: str | None = None,
    dither_strength: float = DITHER_STRENGTH,
) -> QtOklchCurveEditorWindow:
    """Construct the Qt MVP editor without starting the Qt event loop."""
    app, _ = _ensure_qt_application()
    resolved_image_path = resolve_input_image_path(image_path)
    loaded_image = load_image_data(
        resolved_image_path,
        alpha_mask_path=alpha_mask_path,
        mask_seed_points=mask_seed_points,
        mask_seed_point=mask_seed_point,
        mask_color_tolerance=mask_color_tolerance,
        mask_region_offset=mask_region_offset,
    )
    rgb_float = loaded_image.rgb_float
    oklch_float = loaded_image.oklch_float
    valid_mask = loaded_image.valid_mask
    base_model, _ = build_oklch_curve_model(oklch_float, valid_mask)
    curve_overrides = load_state_curve_overrides(curve_path)
    window = QtOklchCurveEditorWindow(
        resolved_image_path,
        rgb_float,
        oklch_float,
        valid_mask,
        base_model,
        initial_curve_overrides=curve_overrides,
        dither_strength=dither_strength,
        curve_output_path=curve_output_path,
    )
    window._qt_application = app
    window.image_warnings = loaded_image.image_warnings
    window.alpha_source = loaded_image.alpha_source
    window.alpha_mask_path = loaded_image.alpha_mask_path
    window.show_warning_dialogs = False
    return _track_top_level_window(window)


def build_qt_editor_launcher(
    *,
    image_path: str | None = None,
    alpha_mask_path: str | None = None,
    curve_path: str | None = None,
    curve_output_path: str | None = None,
    dither_strength: float = DITHER_STRENGTH,
) -> QtEditorLauncherWindow:
    """Construct the Qt launcher without starting the event loop."""
    app, _ = _ensure_qt_application()
    window = QtEditorLauncherWindow(
        image_path=image_path,
        alpha_mask_path=alpha_mask_path,
        curve_path=curve_path,
        curve_output_path=curve_output_path,
        dither_strength=dither_strength,
    )
    window._qt_application = app
    return _track_top_level_window(window)


def launch_qt_editor(
    image_path: str | None,
    *,
    alpha_mask_path: str | None = None,
    mask_seed_points: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    mask_seed_point: tuple[int, int] | None = None,
    mask_color_tolerance: int = DEFAULT_SEED_MASK_COLOR_TOLERANCE,
    mask_region_offset: int = 0,
    curve_path: str | None = None,
    curve_output_path: str | None = None,
    dither_strength: float = DITHER_STRENGTH,
    run_event_loop: bool = True,
    show_warning_dialogs: bool = True,
) -> QtOklchCurveEditorWindow | None:
    """Launch the Qt MVP editor and optionally enter the Qt event loop."""
    app, owns_app = _ensure_qt_application()
    (
        resolved_alpha_mask_path,
        resolved_mask_seed_points,
        resolved_mask_color_tolerance,
        resolved_mask_region_offset,
        prompt_was_shown,
        cancelled,
    ) = _resolve_qt_mask_loading(
        None,
        resolve_input_image_path(image_path),
        alpha_mask_path,
        mask_seed_points,
        mask_seed_point,
        mask_color_tolerance=mask_color_tolerance,
        mask_region_offset=mask_region_offset,
        prompt_user=show_warning_dialogs,
    )
    if cancelled:
        return None
    window = build_qt_editor(
        image_path,
        alpha_mask_path=resolved_alpha_mask_path,
        mask_seed_points=resolved_mask_seed_points,
        mask_color_tolerance=resolved_mask_color_tolerance,
        mask_region_offset=resolved_mask_region_offset,
        curve_path=curve_path,
        curve_output_path=curve_output_path,
        dither_strength=dither_strength,
    )
    window.show_warning_dialogs = show_warning_dialogs and not prompt_was_shown
    window.show()
    if not prompt_was_shown:
        window._show_image_warnings("Input Image Warning", getattr(window, "image_warnings", ()))
    if run_event_loop and owns_app:
        app.exec()
    return window


def launch_qt_editor_launcher(
    *,
    image_path: str | None = None,
    alpha_mask_path: str | None = None,
    curve_path: str | None = None,
    curve_output_path: str | None = None,
    dither_strength: float = DITHER_STRENGTH,
    run_event_loop: bool = True,
) -> QtEditorLauncherWindow:
    """Launch the Qt file-selection launcher and optionally enter the event loop."""
    app, owns_app = _ensure_qt_application()
    window = build_qt_editor_launcher(
        image_path=image_path,
        alpha_mask_path=alpha_mask_path,
        curve_path=curve_path,
        curve_output_path=curve_output_path,
        dither_strength=dither_strength,
    )
    window.show()
    if run_event_loop and owns_app:
        app.exec()
    return window


__all__ = [
    "CURVE_BACKGROUND_HEIGHT",
    "CURVE_BACKGROUND_WIDTH",
    "CURVE_LINE_SAMPLES",
    "CURVE_X_DENSE",
    "QtEditorLauncherWindow",
    "QtOklchCurveEditorWindow",
    "build_qt_editor",
    "build_qt_editor_launcher",
    "launch_qt_editor",
    "launch_qt_editor_launcher",
]