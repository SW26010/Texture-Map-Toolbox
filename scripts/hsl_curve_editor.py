"""
交互式 HLS 曲线编辑器

用法：
    python scripts/hsl_curve_editor.py [图片路径]

操作：
    - 左键拖拽：移动控制点，实时更新预览
    - 左键点击空白处：不操作
    - 右键点击控制点：重置该控制点到默认值（直线位置）
"""

import sys
import os
import time
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io, color
from skimage.util import img_as_float, img_as_ubyte
import colour

# ── 加载图像 ────────────────────────────────────────────────────────────────

def load_image(image_path: str):
    original = io.imread(image_path)
    image_float = img_as_float(original)
    rgb_float = image_float[:, :, :3]
    alpha_float = image_float[:, :, 3] if original.shape[2] == 4 else np.ones(rgb_float.shape[:2])
    valid_mask = alpha_float > 0.5
    luma_float = color.rgb2gray(rgb_float)
    luma_uint8 = np.clip(np.round(luma_float * 255), 0, 255).astype(np.uint8)
    return rgb_float, luma_uint8, valid_mask


def build_base_lut(luma_uint8, rgb_float, valid_mask):
    """构建基础亮度→RGB LUT（归一化浮点，NaN 表示未出现的灰度值）。"""
    valid_luma = luma_uint8[valid_mask]
    valid_rgb = rgb_float[valid_mask]
    histogram = np.bincount(valid_luma, minlength=256)
    rgb_sums = np.stack(
        [np.bincount(valid_luma, weights=ch, minlength=256) for ch in valid_rgb.T], axis=1
    )
    safe_counts = np.maximum(histogram, 1)[:, np.newaxis]
    lut_rgb = rgb_sums / safe_counts
    lut_rgb[histogram == 0] = np.nan
    return lut_rgb  # (256, 3) float, NaN for absent luma values


# ── Spline 曲线工具 ──────────────────────────────────────────────────────────

N_CTRL = 10  # 每条曲线的控制点数
X_CTRL = np.linspace(0, 1, N_CTRL)  # 控制点 x 位置（均匀分布于 [0,1]）
X_256 = np.linspace(0, 1, 256)      # 输出到 256 个灰度值的采样点


def ctrl_to_curve(y_ctrl: np.ndarray) -> np.ndarray:
    """给定 N_CTRL 个控制点 y 值，用 Cubic Spline 插值到 256 点。"""
    cs = CubicSpline(X_CTRL, y_ctrl, bc_type="not-a-knot", extrapolate=True)
    return np.clip(cs(X_256), 0.0, 1.0)


def apply_hsl_curves(lut_rgb, h_curve, l_curve, s_curve):
    """
    将 HLS 调整曲线应用于基础 RGB LUT，返回调整后的 RGB LUT。

    曲线的含义：输出值 = curve[原始值 × 255]（映射关系，非偏移量）。
    NaN 条目保持为 0（透明区域不显示颜色）。
    """
    lut_safe = np.nan_to_num(lut_rgb, nan=0.0)  # (256, 3)
    lut_hsl = colour.RGB_to_HSL(lut_safe)       # (256, 3): H[0,1], S[0,1], L[0,1]

    idx = np.arange(256)

    # 分别重映射 H / S / L
    # h_curve: 输入 H → 输出 H（都在 [0,1]，代表 0-360°）
    h_in = lut_hsl[:, 0]                         # 原始 H [0,1]
    h_idx = np.clip(np.round(h_in * 255).astype(int), 0, 255)
    lut_hsl[:, 0] = h_curve[h_idx]

    l_in = lut_hsl[:, 2]
    l_idx = np.clip(np.round(l_in * 255).astype(int), 0, 255)
    lut_hsl[:, 2] = l_curve[l_idx]

    s_in = lut_hsl[:, 1]
    s_idx = np.clip(np.round(s_in * 255).astype(int), 0, 255)
    lut_hsl[:, 1] = s_curve[s_idx]

    adjusted_rgb = colour.HSL_to_RGB(lut_hsl)
    adjusted_rgb = np.clip(adjusted_rgb, 0.0, 1.0)
    # 恢复 NaN 条目
    absent = np.isnan(lut_rgb[:, 0])
    adjusted_rgb[absent] = 0.0
    return adjusted_rgb  # (256, 3)


def recolor(luma_uint8, lut_rgb_adjusted, valid_mask):
    """用调整后的 LUT 对图像重新上色，返回 uint8 RGB。"""
    recolored = lut_rgb_adjusted[luma_uint8]
    recolored_uint8 = img_as_ubyte(np.clip(recolored, 0.0, 1.0))
    out = np.zeros((*valid_mask.shape, 3), dtype=np.uint8)
    out[valid_mask] = recolored_uint8[valid_mask]
    return out


def recolor_fast(luma_uint8_small, lut_uint8, invalid_mask_small, out_buf):
    """预览专用：LUT 为 uint8，复用预分配的 out_buf 避免每帧堆分配。

    np.take 支持 out= 参数，直接写入已有内存，比每次新建数组快 ~17%。
    """
    np.take(lut_uint8, luma_uint8_small, axis=0, out=out_buf)
    out_buf[invalid_mask_small] = 0


# ── 交互编辑器 ───────────────────────────────────────────────────────────────

class HslCurveEditor:
    CURVE_CFG = [
        {"name": "Hue",        "color": "orchid",      "ylim": (0, 1), "ylabel": "Output H (0=0°, 1=360°)"},
        {"name": "Lightness",  "color": "gold",        "ylim": (0, 1), "ylabel": "Output L"},
        {"name": "Saturation", "color": "deepskyblue", "ylim": (0, 1), "ylabel": "Output S"},
    ]

    def __init__(self, rgb_float, luma_uint8, valid_mask, lut_rgb):
        self.rgb_float = rgb_float
        self.luma_uint8 = luma_uint8
        self.valid_mask = valid_mask
        self.lut_rgb = lut_rgb

        # 预览用缩小版（25% 面积，即 50% 线性尺寸）
        PREVIEW_SCALE = 0.25
        h, w = luma_uint8.shape
        ph, pw = max(1, int(h * PREVIEW_SCALE)), max(1, int(w * PREVIEW_SCALE))
        # 用切片降采样（速度快，无需额外库）
        sy = max(1, h // ph)
        sx = max(1, w // pw)
        self.luma_small = luma_uint8[::sy, ::sx]
        self.mask_small = valid_mask[::sy, ::sx]
        # 预分配预览输出缓冲区（H×W×3 uint8），每帧复用，避免堆分配
        self._preview_buf = np.empty((*self.luma_small.shape, 3), dtype=np.uint8)
        # 预计算反模，避免每帧重新计算 ~mask
        self._invalid_mask = ~self.mask_small

        # 默认控制点：恒等映射（对角线）
        self.ctrl_y = [np.copy(X_CTRL) for _ in range(3)]
        self.curves = [ctrl_to_curve(cy) for cy in self.ctrl_y]

        self._drag_state = None  # (curve_idx, point_idx)

        self._build_ui()
        self._render_preview()
        self._connect_events()

    def _build_ui(self):
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title("HLS Curve Editor")

        # 左侧 3 个曲线编辑子图，右侧 1 个预览图
        gs = self.fig.add_gridspec(3, 2, width_ratios=[1, 1.2], hspace=0.4, wspace=0.35,
                                   left=0.07, right=0.97, top=0.95, bottom=0.07)
        self.curve_axes = [self.fig.add_subplot(gs[i, 0]) for i in range(3)]
        self.preview_ax = self.fig.add_subplot(gs[:, 1])

        self.curve_lines = []
        self.ctrl_scatters = []

        for i, (ax, cfg) in enumerate(zip(self.curve_axes, self.CURVE_CFG)):
            ax.set_xlim(0, 1)
            ax.set_ylim(*cfg["ylim"])
            ax.set_title(cfg["name"], fontsize=10)
            ax.set_ylabel(cfg["ylabel"], fontsize=8)
            ax.grid(True, alpha=0.3)

            line, = ax.plot(X_256, self.curves[i], color=cfg["color"], lw=2)
            scat = ax.scatter(X_CTRL, self.ctrl_y[i], s=60, color="white",
                              edgecolors=cfg["color"], zorder=5, picker=True)
            # 对角参考线
            ax.plot([0, 1], [0, 1], color="gray", lw=0.8, linestyle="--", alpha=0.5)

            self.curve_lines.append(line)
            self.ctrl_scatters.append(scat)

        self.preview_ax.set_title("Preview", fontsize=11)
        self.preview_ax.axis("off")
        self.preview_img = self.preview_ax.imshow(
            np.zeros((*self.valid_mask.shape, 3), dtype=np.uint8)
        )

    def _render_preview(self):
        h_curve, l_curve, s_curve = self.curves

        t0 = time.perf_counter()
        lut_adjusted_float = apply_hsl_curves(self.lut_rgb, h_curve, l_curve, s_curve)
        # 量化为 uint8 LUT（256×3），后续查表无需类型转换
        lut_uint8 = np.clip(np.round(lut_adjusted_float * 255), 0, 255).astype(np.uint8)
        t1 = time.perf_counter()
        recolor_fast(self.luma_small, lut_uint8, self._invalid_mask, self._preview_buf)
        t2 = time.perf_counter()
        self.preview_img.set_data(self._preview_buf)
        self.fig.canvas.draw_idle()
        t3 = time.perf_counter()

        print(
            f"apply+quantize={1000*(t1-t0):.1f}ms  "
            f"recolor={1000*(t2-t1):.1f}ms  "
            f"draw={1000*(t3-t2):.1f}ms  "
            f"total={1000*(t3-t0):.1f}ms"
        )

    def _update_curve(self, curve_idx):
        cy = self.ctrl_y[curve_idx]
        self.curves[curve_idx] = ctrl_to_curve(cy)
        self.curve_lines[curve_idx].set_ydata(self.curves[curve_idx])
        # 更新散点位置
        offsets = np.column_stack([X_CTRL, cy])
        self.ctrl_scatters[curve_idx].set_offsets(offsets)

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def _hit_test(self, event):
        """返回被点击的 (curve_idx, point_idx) 或 None。"""
        for i, ax in enumerate(self.curve_axes):
            if event.inaxes is not ax:
                continue
            if event.xdata is None or event.ydata is None:
                continue
            # 转换为 axes 坐标系下的像素距离
            ax_bbox = ax.get_window_extent()
            ax_width = ax_bbox.width   # pixels
            ax_height = ax_bbox.height
            x_range = ax.get_xlim()
            y_range = ax.get_ylim()

            def to_px(x, y):
                px = (x - x_range[0]) / (x_range[1] - x_range[0]) * ax_width
                py = (y - y_range[0]) / (y_range[1] - y_range[0]) * ax_height
                return px, py

            ex, ey = to_px(event.xdata, event.ydata)
            for j, (cx, cy) in enumerate(zip(X_CTRL, self.ctrl_y[i])):
                cpx, cpy = to_px(cx, cy)
                if (ex - cpx) ** 2 + (ey - cpy) ** 2 < 12 ** 2:
                    return i, j
        return None

    def _on_press(self, event):
        hit = self._hit_test(event)
        if hit is None:
            return
        ci, pi = hit
        if event.button == 3:  # 右键重置
            x_default = X_CTRL[pi]
            self.ctrl_y[ci][pi] = x_default
            self._update_curve(ci)
            self._render_preview()
        elif event.button == 1:
            self._drag_state = (ci, pi)

    def _on_release(self, event):
        self._drag_state = None

    def _on_motion(self, event):
        if self._drag_state is None:
            return
        ci, pi = self._drag_state
        ax = self.curve_axes[ci]
        if event.inaxes is not ax:
            return
        if event.ydata is None:
            return
        ylim = self.CURVE_CFG[ci]["ylim"]
        new_y = float(np.clip(event.ydata, ylim[0], ylim[1]))
        self.ctrl_y[ci][pi] = new_y
        self._update_curve(ci)
        self._render_preview()

    def show(self):
        plt.show()


# ── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = os.path.join(os.path.dirname(__file__), "..", "data", "mtmtPonyTail.png")

    print(f"Loading: {image_path}")
    rgb_float, luma_uint8, valid_mask = load_image(image_path)
    lut_rgb = build_base_lut(luma_uint8, rgb_float, valid_mask)
    print("Building LUT done. Opening editor...")

    editor = HslCurveEditor(rgb_float, luma_uint8, valid_mask, lut_rgb)
    editor.show()
