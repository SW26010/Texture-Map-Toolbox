import argparse
from dataclasses import dataclass
import json
import os
import sys

import colour
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from skimage import io
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import img_as_float, img_as_ubyte


MIN_KEYPOINTS = 256
HUE_CHROMA_FLOOR = 1e-4
DITHER_STRENGTH = 0.0
DITHER_SEED = 0
STATE_CURVE_CTRL_POINTS = 16


@dataclass
class OklchCurveModel:
    key_y: np.ndarray
    key_c: np.ndarray
    key_h: np.ndarray
    c_interp: PchipInterpolator
    u_interp: PchipInterpolator
    v_interp: PchipInterpolator


@dataclass
class StateCurveSet:
    lightness_points: np.ndarray
    chroma_points: np.ndarray
    hue_points: np.ndarray
    lightness_interp: PchipInterpolator
    chroma_interp: PchipInterpolator
    hue_u_interp: PchipInterpolator
    hue_v_interp: PchipInterpolator


def rgb_to_oklch(rgb_float: np.ndarray) -> np.ndarray:
    """将 sRGB 浮点图像转换为 Oklch。"""
    return colour.Oklab_to_Oklch(colour.XYZ_to_Oklab(colour.sRGB_to_XYZ(rgb_float)))


def oklch_to_rgb(oklch_float: np.ndarray) -> np.ndarray:
    """将 Oklch 浮点图像转换回 sRGB。"""
    return colour.XYZ_to_sRGB(colour.Oklab_to_XYZ(colour.Oklch_to_Oklab(oklch_float)))


def load_image(image_path: str):
    """加载图像，提取 RGB、Alpha 和原始 Oklch。"""
    original_image = io.imread(image_path)
    image_float = img_as_float(original_image)
    rgb_float = image_float[:, :, :3]
    alpha_float = (
        image_float[:, :, 3]
        if original_image.shape[2] == 4
        else np.ones(rgb_float.shape[:2], dtype=rgb_float.dtype)
    )
    valid_mask = alpha_float > 0.5
    oklch_float = rgb_to_oklch(rgb_float)
    return rgb_float, oklch_float, valid_mask


def extract_quantile_keypoints(y_samples: np.ndarray, min_keypoints: int) -> np.ndarray:
    """按分位数提取关键点，并补齐端点。"""
    quantiles = np.linspace(0.0, 1.0, max(2, min_keypoints))
    key_y = np.quantile(y_samples, quantiles)
    key_y = np.unique(np.concatenate(([0.0], key_y, [1.0])))
    if key_y.size < min_keypoints:
        unique_samples = np.unique(y_samples)
        fill_indices = np.linspace(0, unique_samples.size - 1, max(2, min_keypoints), dtype=int)
        sampled_fill = unique_samples[fill_indices]
        key_y = np.unique(np.concatenate((key_y, sampled_fill)))
    if key_y.size < 2:
        key_y = np.array([0.0, 1.0], dtype=np.float64)
    return key_y


def aggregate_keypoint_samples(
    y_samples: np.ndarray,
    c_samples: np.ndarray,
    h_samples: np.ndarray,
    key_y: np.ndarray,
    chroma_floor: float,
):
    """对每个关键点聚合代表色度和 hue 向量。"""
    order = np.argsort(y_samples)
    y_sorted = y_samples[order]
    c_sorted = c_samples[order]
    h_sorted = h_samples[order]

    h_rad = np.deg2rad(h_sorted)
    u_sorted = np.cos(h_rad)
    v_sorted = np.sin(h_rad)

    boundaries = np.empty(key_y.size + 1, dtype=np.float64)
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf
    boundaries[1:-1] = 0.5 * (key_y[:-1] + key_y[1:])

    starts = np.searchsorted(y_sorted, boundaries[:-1], side="left")
    ends = np.searchsorted(y_sorted, boundaries[1:], side="left")

    key_c = np.empty_like(key_y)
    key_u = np.empty_like(key_y)
    key_v = np.empty_like(key_y)

    for idx, (start, end) in enumerate(zip(starts, ends)):
        if end <= start:
            center = int(np.clip(np.searchsorted(y_sorted, key_y[idx]), 0, y_sorted.size - 1))
            start = max(0, center - 1)
            end = min(y_sorted.size, center + 2)

        c_window = c_sorted[start:end]
        key_c[idx] = np.median(c_window)

        hue_mask = c_window >= chroma_floor
        if np.any(hue_mask):
            u_value = np.mean(u_sorted[start:end][hue_mask])
            v_value = np.mean(v_sorted[start:end][hue_mask])
        elif idx > 0:
            u_value = key_u[idx - 1]
            v_value = key_v[idx - 1]
        else:
            u_value = 1.0
            v_value = 0.0

        norm = np.hypot(u_value, v_value)
        if norm < 1e-8:
            u_value, v_value = 1.0, 0.0
            norm = 1.0

        key_u[idx] = u_value / norm
        key_v[idx] = v_value / norm

    key_h = (np.degrees(np.arctan2(key_v, key_u)) + 360.0) % 360.0
    return key_c, key_h, key_u, key_v


def build_oklch_curve_model(
    oklch_float: np.ndarray,
    valid_mask: np.ndarray,
    min_keypoints: int = MIN_KEYPOINTS,
    chroma_floor: float = HUE_CHROMA_FLOOR,
):
    """基于原始 Oklch 样本云构建 C(y) / h(y) 连续模型。"""
    valid_oklch = oklch_float[valid_mask]
    y_samples = valid_oklch[:, 0]
    c_samples = valid_oklch[:, 1]
    h_samples = valid_oklch[:, 2]

    key_y = extract_quantile_keypoints(y_samples, min_keypoints)
    key_c, key_h, key_u, key_v = aggregate_keypoint_samples(
        y_samples, c_samples, h_samples, key_y, chroma_floor
    )

    model = OklchCurveModel(
        key_y=key_y,
        key_c=key_c,
        key_h=key_h,
        c_interp=PchipInterpolator(key_y, key_c, extrapolate=True),
        u_interp=PchipInterpolator(key_y, key_u, extrapolate=True),
        v_interp=PchipInterpolator(key_y, key_v, extrapolate=True),
    )
    return model, y_samples


def prepare_control_points(
    control_points: np.ndarray | None,
    default_x: np.ndarray,
    default_y: np.ndarray,
    *,
    clip_min: float | None = None,
    clip_max: float | None = None,
    wrap_degrees: bool = False,
) -> np.ndarray:
    """规范化用户控制点；未提供时回退到默认控制点。"""
    if control_points is None:
        points = np.column_stack([default_x, default_y])
    else:
        points = np.asarray(control_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("control_points must be an (N, 2) array")
        points = points[np.argsort(points[:, 0])]

    points = np.array(points, copy=True)
    points[:, 0] = np.clip(points[:, 0], 0.0, 1.0)
    if wrap_degrees:
        points[:, 1] = np.mod(points[:, 1], 360.0)
    else:
        if clip_min is not None:
            points[:, 1] = np.maximum(points[:, 1], clip_min)
        if clip_max is not None:
            points[:, 1] = np.minimum(points[:, 1], clip_max)

    _, unique_indices = np.unique(points[:, 0], return_index=True)
    points = points[np.sort(unique_indices)]
    if points.shape[0] < 2:
        raise ValueError("control_points must contain at least two unique x positions")
    return points


def build_state_curve_set(
    base_model: OklchCurveModel,
    control_point_count: int = STATE_CURVE_CTRL_POINTS,
    lightness_control_points: np.ndarray | None = None,
    chroma_control_points: np.ndarray | None = None,
    hue_control_points: np.ndarray | None = None,
) -> StateCurveSet:
    """构建用户状态曲线，默认从基础模型采样得到恒等/基线控制点。"""
    default_lightness_x = np.linspace(0.0, 1.0, max(2, control_point_count))

    lightness_points = prepare_control_points(
        lightness_control_points,
        default_lightness_x,
        default_lightness_x,
        clip_min=0.0,
        clip_max=1.0,
    )
    chroma_points = prepare_control_points(
        chroma_control_points,
        base_model.key_y,
        base_model.key_c,
        clip_min=0.0,
    )
    hue_points = prepare_control_points(
        hue_control_points,
        base_model.key_y,
        base_model.key_h,
        wrap_degrees=True,
    )

    hue_rad = np.deg2rad(hue_points[:, 1])
    hue_u = np.cos(hue_rad)
    hue_v = np.sin(hue_rad)

    return StateCurveSet(
        lightness_points=lightness_points,
        chroma_points=chroma_points,
        hue_points=hue_points,
        lightness_interp=PchipInterpolator(lightness_points[:, 0], lightness_points[:, 1], extrapolate=True),
        chroma_interp=PchipInterpolator(chroma_points[:, 0], chroma_points[:, 1], extrapolate=True),
        hue_u_interp=PchipInterpolator(hue_points[:, 0], hue_u, extrapolate=True),
        hue_v_interp=PchipInterpolator(hue_points[:, 0], hue_v, extrapolate=True),
    )


def load_state_curve_overrides(curve_path: str | None) -> dict:
    """从 JSON 文件加载 Lt / Ct / ht 控制点。"""
    if curve_path is None:
        return {}

    with open(curve_path, "r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("curve file must contain a JSON object")

    overrides = {}
    for json_key, arg_key in (
        ("lightness", "lightness_control_points"),
        ("chroma", "chroma_control_points"),
        ("hue", "hue_control_points"),
    ):
        points = payload.get(json_key)
        if points is not None:
            overrides[arg_key] = np.asarray(points, dtype=np.float64)
    return overrides


def apply_precurve_dither(
    y_image: np.ndarray,
    valid_mask: np.ndarray,
    strength: float,
    random_seed: int = DITHER_SEED,
) -> np.ndarray:
    """在用户曲线求值前对输入骨架 y 做可选抖动。"""
    y_dithered = np.array(y_image, copy=True)
    if strength <= 0.0:
        return y_dithered

    rng = np.random.default_rng(random_seed)
    noise = rng.uniform(-strength, strength, size=y_image.shape)
    y_dithered[valid_mask] = np.clip(y_dithered[valid_mask] + noise[valid_mask], 0.0, 1.0)
    return y_dithered


def evaluate_chroma_hue(model: OklchCurveModel, y_values: np.ndarray):
    """对任意 y 求值，返回对应的色度和 hue。"""
    y_clipped = np.clip(y_values, 0.0, 1.0)
    chroma = np.clip(model.c_interp(y_clipped), 0.0, None)
    u_values = model.u_interp(y_clipped)
    v_values = model.v_interp(y_clipped)
    norm = np.hypot(u_values, v_values)
    safe_norm = np.where(norm < 1e-8, 1.0, norm)
    hue = (np.degrees(np.arctan2(v_values / safe_norm, u_values / safe_norm)) + 360.0) % 360.0
    return chroma, hue


def evaluate_state_curves(state_curves: StateCurveSet, y_values: np.ndarray):
    """按用户状态曲线求值得到最终的 L' / C' / h'。"""
    y_clipped = np.clip(y_values, 0.0, 1.0)
    lightness = np.clip(state_curves.lightness_interp(y_clipped), 0.0, 1.0)
    chroma = np.clip(state_curves.chroma_interp(lightness), 0.0, None)

    hue_u = state_curves.hue_u_interp(lightness)
    hue_v = state_curves.hue_v_interp(lightness)
    norm = np.hypot(hue_u, hue_v)
    safe_norm = np.where(norm < 1e-8, 1.0, norm)
    hue = (np.degrees(np.arctan2(hue_v / safe_norm, hue_u / safe_norm)) + 360.0) % 360.0
    return lightness, chroma, hue


def compress_oklch_chroma_to_srgb(
    oklch_values: np.ndarray,
    valid_mask: np.ndarray | None = None,
    iterations: int = 12,
):
    """固定 L 和 h，仅通过压缩 C 将颜色拉回 sRGB gamut。"""
    if valid_mask is None:
        valid_mask = np.ones(oklch_values.shape[:-1], dtype=bool)

    rgb_float = oklch_to_rgb(oklch_values)
    in_gamut = np.all((rgb_float >= 0.0) & (rgb_float <= 1.0), axis=-1)
    invalid_mask = valid_mask & ~in_gamut
    if not np.any(invalid_mask):
        rgb_float[~valid_mask] = 0.0
        return oklch_values, np.clip(rgb_float, 0.0, 1.0), 0

    invalid_oklch = oklch_values[invalid_mask]
    lightness = invalid_oklch[:, 0]
    chroma_hi = invalid_oklch[:, 1]
    hue = invalid_oklch[:, 2]
    chroma_lo = np.zeros_like(chroma_hi)

    for _ in range(iterations):
        chroma_mid = 0.5 * (chroma_lo + chroma_hi)
        candidate_oklch = np.column_stack([lightness, chroma_mid, hue])
        candidate_rgb = oklch_to_rgb(candidate_oklch)
        candidate_ok = np.all((candidate_rgb >= 0.0) & (candidate_rgb <= 1.0), axis=1)
        chroma_lo = np.where(candidate_ok, chroma_mid, chroma_lo)
        chroma_hi = np.where(candidate_ok, chroma_hi, chroma_mid)

    adjusted_oklch = np.array(oklch_values, copy=True)
    adjusted_invalid = adjusted_oklch[invalid_mask]
    adjusted_invalid[:, 1] = chroma_lo
    adjusted_oklch[invalid_mask] = adjusted_invalid

    adjusted_rgb = np.clip(oklch_to_rgb(adjusted_oklch), 0.0, 1.0)
    adjusted_rgb[~valid_mask] = 0.0
    return adjusted_oklch, adjusted_rgb, int(np.count_nonzero(invalid_mask))


def reconstruct_from_state_curves(
    oklch_float: np.ndarray,
    valid_mask: np.ndarray,
    state_curves: StateCurveSet,
    dither_strength: float = DITHER_STRENGTH,
):
    """对抖动后的输入轴先求 Lt，再通过 Ct / ht 生成最终颜色状态。"""
    y_image = oklch_float[:, :, 0]
    y_eval = apply_precurve_dither(y_image, valid_mask, dither_strength)
    lightness_eval, chroma_eval, hue_eval = evaluate_state_curves(state_curves, y_eval)
    reconstructed_oklch = np.stack([lightness_eval, chroma_eval, hue_eval], axis=-1)
    reconstructed_oklch[~valid_mask] = 0.0
    adjusted_oklch, reconstructed_rgb, gamut_compressed_pixels = compress_oklch_chroma_to_srgb(
        reconstructed_oklch, valid_mask
    )
    return reconstructed_rgb, adjusted_oklch, y_eval, gamut_compressed_pixels


def evaluate_reconstruction(rgb_float, recolored_rgb_float, valid_mask):
    """用 PSNR 和 Delta E 2000 评估重建效果。"""
    recolored_rgb_int = img_as_ubyte(np.clip(recolored_rgb_float, 0.0, 1.0))
    recolored_quantized_float = img_as_float(recolored_rgb_int)

    psnr = peak_signal_noise_ratio(
        img_as_ubyte(rgb_float)[valid_mask], recolored_rgb_int[valid_mask]
    )

    valid_recolored = recolored_quantized_float[valid_mask]
    valid_original = rgb_float[valid_mask]

    delta_e_valid = colour.difference.delta_E_CIE2000(
        colour.XYZ_to_Lab(colour.sRGB_to_XYZ(valid_recolored)),
        colour.XYZ_to_Lab(colour.sRGB_to_XYZ(valid_original)),
    )

    delta_e_image = np.zeros(valid_mask.shape, dtype=np.float64)
    delta_e_image[valid_mask] = delta_e_valid
    delta_e_stats = {
        "mean": np.mean(delta_e_valid),
        "median": np.median(delta_e_valid),
        "std_dev": np.std(delta_e_valid),
        "max": np.max(delta_e_valid),
        "95th_percentile": np.percentile(delta_e_valid, 95),
    }
    return recolored_rgb_int, psnr, delta_e_image, delta_e_stats


def plot_comparison(rgb_float, y_image, recolored_rgb_int, valid_mask, psnr, delta_e_image):
    """绘制输入轴、原图、重建图和色差图。"""
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    ax[0].imshow(np.where(valid_mask, y_image, 0.0), cmap="gray", vmin=0.0, vmax=1.0)
    ax[0].set_title("Input Oklch Lightness (L0)")

    ax[1].imshow(np.where(valid_mask[..., np.newaxis], img_as_ubyte(rgb_float), 0))
    ax[1].set_title("Original Image")

    ax[2].imshow(np.where(valid_mask[..., np.newaxis], recolored_rgb_int, 0))
    ax[2].set_title(f"Re-colored from Oklch L0\nPSNR = {psnr:.2f} dB")

    delta_e_max = 2.0
    delta_e_clipped = np.clip(delta_e_image, 0.0, delta_e_max)
    im_de = ax[3].imshow(
        np.where(valid_mask, delta_e_clipped, 0.0),
        cmap="gray",
        vmin=0.0,
        vmax=delta_e_max,
    )
    ax[3].set_title(f"CIEDE2000 Error Map (0-{delta_e_max})")
    pos3 = ax[3].get_position()
    cax = fig.add_axes([pos3.x1 + 0.005, pos3.y0, 0.015, pos3.height])
    cbar = fig.colorbar(im_de, cax=cax)
    cbar.set_label("Delta E 2000 Value")

    for axis in ax:
        axis.axis("off")
    plt.show(block=False)


def plot_analysis(y_samples: np.ndarray, model: OklchCurveModel):
    """绘制 Oklch 输入轴分布、C(y)、h(y) 和 LUT 预览。"""
    histogram, _ = np.histogram(y_samples, bins=256, range=(0.0, 1.0))
    histogram_for_plot = np.maximum(histogram, 1)
    x_hist = np.linspace(0.0, 1.0, 256, endpoint=False)
    x_dense = np.linspace(0.0, 1.0, 512)
    chroma_dense, hue_dense = evaluate_chroma_hue(model, x_dense)

    lut_oklch = np.column_stack([x_dense, chroma_dense, hue_dense])
    _, lut_rgb, _ = compress_oklch_chroma_to_srgb(lut_oklch)

    fig, ax = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    ax[0].bar(x_hist, histogram_for_plot, width=1.0 / 256.0, align="edge", color="royalblue", alpha=0.8)
    ax[0].set_yscale("log")
    ax[0].set_title("Input Oklch Lightness Histogram")
    ax[0].grid(axis="y", linestyle="--", alpha=0.6)
    ax[0].set_xlim(0.0, 1.0)

    ax[1].plot(x_dense, chroma_dense, color="deepskyblue", linewidth=2)
    ax[1].scatter(model.key_y, model.key_c, s=14, color="black", alpha=0.65)
    ax[1].set_title("Fitted Chroma Curve C(y)")
    ax[1].grid(axis="y", linestyle="--", alpha=0.6)

    ax[2].imshow(
        np.broadcast_to(lut_rgb[np.newaxis, :, :], (32, lut_rgb.shape[0], 3)),
        aspect="auto",
        extent=[0.0, 1.0, 0.0, 1.0],
    )
    ax[2].set_title("Oklch LUT Preview")
    ax[2].set_yticks([])

    ax[3].plot(x_dense, hue_dense, color="black", linewidth=1.5)
    ax[3].scatter(model.key_y, model.key_h, s=14, color="tomato", alpha=0.7)
    ax[3].set_ylim(0.0, 360.0)
    ax[3].set_title("Fitted Hue Curve h(y)")
    ax[3].grid(axis="y", linestyle="--", alpha=0.6)
    ax[3].set_xlabel("Input Oklch Lightness (L0)")

    plt.tight_layout()
    plt.show(block=False)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Build and evaluate an Oklch-based luma color map.")
    parser.add_argument(
        "image_path",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "mtmtPonyTail.png"),
        help="Input image path.",
    )
    parser.add_argument(
        "--curves",
        dest="curve_path",
        help="Optional JSON file containing lightness/chroma/hue control points.",
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
    image_path = args.image_path

    rgb_float, oklch_float, valid_mask = load_image(image_path)
    model, y_samples = build_oklch_curve_model(oklch_float, valid_mask)
    curve_overrides = load_state_curve_overrides(args.curve_path)
    state_curves = build_state_curve_set(model, **curve_overrides)
    recolored_rgb_float, _, y_eval, gamut_compressed_pixels = reconstruct_from_state_curves(
        oklch_float, valid_mask, state_curves, dither_strength=args.dither_strength
    )
    recolored_rgb_int, psnr, delta_e_image, delta_e_stats = evaluate_reconstruction(
        rgb_float, recolored_rgb_float, valid_mask
    )

    print("Input axis: original Oklch Lightness (L0)")
    print(f"Curve source: {args.curve_path or 'default base-model controls'}")
    print(f"Quantile keypoints: {model.key_y.size}")
    print(
        "State-curve control points: "
        f"L={state_curves.lightness_points.shape[0]}, "
        f"C={state_curves.chroma_points.shape[0]}, "
        f"h={state_curves.hue_points.shape[0]}"
    )
    print(f"Pre-curve dither strength: {args.dither_strength:.6f}")
    print(f"Gamut-compressed pixels: {gamut_compressed_pixels}")
    print(f"PSNR (Peak Signal-to-Noise Ratio): {psnr:.2f} dB")
    for key, value in delta_e_stats.items():
        print(f"Delta E 2000 ({key.replace('_', ' ').title()}): {value:.2f}")

    plot_comparison(rgb_float, y_eval, recolored_rgb_int, valid_mask, psnr, delta_e_image)
    plot_analysis(y_samples, model)
    plt.show()
