"""Core Oklch luma workflow implementation."""

from dataclasses import dataclass
import json
from pathlib import Path

import colour
import numpy as np
from scipy.ndimage import binary_propagation
from scipy.interpolate import PchipInterpolator
from skimage import io
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import img_as_float, img_as_ubyte


MIN_KEYPOINTS = 256
HUE_CHROMA_FLOOR = 1e-4
DITHER_STRENGTH = 0.0
DITHER_SEED = 0
STATE_CURVE_CTRL_POINTS = 16
DEFAULT_FAST_PREVIEW_SCALE = 0.25
DEFAULT_FAST_LUT_SIZE = 512
SUPPORTED_LUMA_ALGORITHMS = ("original", "fast")
DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "data"
DEFAULT_SAMPLE_IMAGE_CANDIDATES = (
    DATA_DIRECTORY / "mtmtPonyTail_custom.png",
    DATA_DIRECTORY / "mtmtPonyTail.png",
)
SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
JPEG_IMAGE_SUFFIXES = (".jpg", ".jpeg")
ALPHA_VALID_EPSILON = 1e-6
AUTO_MASK_BORDER_WIDTH = 2
AUTO_MASK_COLOR_TOLERANCE = 6
AUTO_MASK_HISTOGRAM_BIN_SIZE = 8
AUTO_MASK_MAX_BORDER_STD = 6.0
AUTO_MASK_MIN_EDGE_RUN = 3
AUTO_MASK_MIN_BORDER_PIXELS = 8


def _iter_default_sample_image_candidates():
    """Yield preferred bundled samples, then any other supported image in the data directory."""
    seen_paths: set[Path] = set()
    for candidate in DEFAULT_SAMPLE_IMAGE_CANDIDATES:
        if candidate not in seen_paths:
            seen_paths.add(candidate)
            yield candidate

    if not DATA_DIRECTORY.exists():
        return

    for candidate in sorted(DATA_DIRECTORY.iterdir()):
        if candidate in seen_paths:
            continue
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            seen_paths.add(candidate)
            yield candidate


def resolve_input_image_path(image_path: str | None) -> str:
    """解析输入图路径；若未提供则回退到仓库内可用样例图。"""
    if image_path:
        resolved_path = Path(image_path).expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"input image not found: {resolved_path}")
        return str(resolved_path)

    for candidate in _iter_default_sample_image_candidates():
        if candidate.exists():
            return str(candidate)

    raise ValueError("image_path is required because no bundled sample image is available in this checkout")


def resolve_optional_path(path: str | None, *, argument_name: str) -> str | None:
    """Resolve an optional existing path without falling back to bundled samples."""
    if path is None:
        return None
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"{argument_name} not found: {resolved_path}")
    return str(resolved_path)


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


@dataclass
class LumaExecutionRequest:
    image_path: str | None = None
    alpha_mask_path: str | None = None
    curve_path: str | None = None
    algorithm: str = "original"
    dither_strength: float = DITHER_STRENGTH
    evaluate_result: bool = True
    show_plots: bool = True
    preview_scale: float = DEFAULT_FAST_PREVIEW_SCALE
    preview_lut_size: int = DEFAULT_FAST_LUT_SIZE
    output_image_path: str | None = None
    result_json_path: str | None = None


@dataclass
class LumaPreviewFrame:
    rgb_float: np.ndarray
    oklch_float: np.ndarray
    valid_mask: np.ndarray
    y_image: np.ndarray
    scale: float


@dataclass
class LoadedImageData:
    image_path: str
    alpha_mask_path: str | None
    rgb_float: np.ndarray
    alpha_float: np.ndarray
    oklch_float: np.ndarray
    valid_mask: np.ndarray
    alpha_source: str
    image_warnings: tuple[str, ...] = ()
    mask_prompt_required: bool = False


@dataclass
class LumaExecutionResult:
    algorithm: str
    image_path: str
    alpha_mask_path: str | None
    curve_path: str | None
    dither_strength: float
    source_image_shape: tuple[int, int]
    output_image_shape: tuple[int, int]
    output_scale: float
    preview_lut_size: int | None
    rgb_float: np.ndarray
    oklch_float: np.ndarray
    valid_mask: np.ndarray
    model: OklchCurveModel
    y_samples: np.ndarray
    state_curves: StateCurveSet
    recolored_rgb_float: np.ndarray
    y_eval: np.ndarray
    gamut_compressed_pixels: int
    alpha_source: str = "embedded"
    image_warnings: tuple[str, ...] = ()
    recolored_rgb_int: np.ndarray | None = None
    psnr: float | None = None
    delta_e_image: np.ndarray | None = None
    delta_e_stats: dict[str, float] | None = None
    gamut_compressed_lut_entries: int | None = None
    output_image_path: str | None = None


def rgb_to_oklch(rgb_float: np.ndarray) -> np.ndarray:
    """将 sRGB 浮点图像转换为 Oklch。"""
    linear_rgb = srgb_to_linear(rgb_float)
    oklab = linear_srgb_to_oklab(linear_rgb)
    lightness = oklab[:, :, 0]
    a_channel = oklab[:, :, 1]
    b_channel = oklab[:, :, 2]
    chroma = np.hypot(a_channel, b_channel)
    hue = (np.degrees(np.arctan2(b_channel, a_channel)) + 360.0) % 360.0
    return np.stack([lightness, chroma, hue], axis=-1)


def oklch_to_rgb(oklch_float: np.ndarray) -> np.ndarray:
    """将 Oklch 浮点图像转换回 sRGB。"""
    lightness = oklch_float[..., 0]
    chroma = oklch_float[..., 1]
    hue_rad = np.deg2rad(oklch_float[..., 2])
    oklab = np.stack([lightness, chroma * np.cos(hue_rad), chroma * np.sin(hue_rad)], axis=-1)
    linear_rgb = oklab_to_linear_srgb(oklab)
    return linear_to_srgb(linear_rgb)


def srgb_to_linear(rgb_float: np.ndarray) -> np.ndarray:
    """将 gamma-encoded sRGB 转换为 linear sRGB。"""
    rgb_float = np.asarray(rgb_float, dtype=np.float64)
    linear_rgb = np.empty_like(rgb_float)
    low_mask = rgb_float <= 0.04045
    linear_rgb[low_mask] = rgb_float[low_mask] / 12.92
    high_mask = ~low_mask
    linear_rgb[high_mask] = ((rgb_float[high_mask] + 0.055) / 1.055) ** 2.4
    return linear_rgb


def linear_to_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    """将 linear sRGB 转换为 gamma-encoded sRGB，并保留越界符号供 gamut 检测。"""
    linear_rgb = np.asarray(linear_rgb, dtype=np.float64)
    rgb_float = np.empty_like(linear_rgb)
    low_mask = linear_rgb <= 0.0031308
    rgb_float[low_mask] = 12.92 * linear_rgb[low_mask]
    high_mask = ~low_mask
    rgb_float[high_mask] = 1.055 * np.power(linear_rgb[high_mask], 1.0 / 2.4) - 0.055
    return rgb_float


def linear_srgb_to_oklab(linear_rgb: np.ndarray) -> np.ndarray:
    """按 Oklab 原始公式将 linear sRGB 转为 Oklab。"""
    red = linear_rgb[..., 0]
    green = linear_rgb[..., 1]
    blue = linear_rgb[..., 2]

    l_channel = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    m_channel = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    s_channel = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue

    l_root = np.cbrt(l_channel)
    m_root = np.cbrt(m_channel)
    s_root = np.cbrt(s_channel)

    return np.stack(
        [
            0.2104542553 * l_root + 0.7936177850 * m_root - 0.0040720468 * s_root,
            1.9779984951 * l_root - 2.4285922050 * m_root + 0.4505937099 * s_root,
            0.0259040371 * l_root + 0.7827717662 * m_root - 0.8086757660 * s_root,
        ],
        axis=-1,
    )


def oklab_to_linear_srgb(oklab: np.ndarray) -> np.ndarray:
    """按 Oklab 原始公式将 Oklab 转回 linear sRGB。"""
    lightness = oklab[..., 0]
    a_channel = oklab[..., 1]
    b_channel = oklab[..., 2]

    l_root = lightness + 0.3963377774 * a_channel + 0.2158037573 * b_channel
    m_root = lightness - 0.1055613458 * a_channel - 0.0638541728 * b_channel
    s_root = lightness - 0.0894841775 * a_channel - 1.2914855480 * b_channel

    l_channel = l_root * l_root * l_root
    m_channel = m_root * m_root * m_root
    s_channel = s_root * s_root * s_root

    return np.stack(
        [
            4.0767416621 * l_channel - 3.3077115913 * m_channel + 0.2309699292 * s_channel,
            -1.2684380046 * l_channel + 2.6097574011 * m_channel - 0.3413193965 * s_channel,
            -0.0041960863 * l_channel - 0.7034186147 * m_channel + 1.7076147010 * s_channel,
        ],
        axis=-1,
    )


def compute_oklab_max_saturation(a_unit: np.ndarray, b_unit: np.ndarray) -> np.ndarray:
    """求给定 hue 方向在 sRGB 内可达到的最大 S=C/L。"""
    red_region = (-1.88170328 * a_unit - 0.80936493 * b_unit) > 1.0
    green_region = (~red_region) & ((1.81444104 * a_unit - 1.19445276 * b_unit) > 1.0)
    blue_region = ~(red_region | green_region)

    k0 = np.select([red_region, green_region, blue_region], [1.19086277, 0.73956515, 1.35733652])
    k1 = np.select([red_region, green_region, blue_region], [1.76576728, -0.45954404, -0.00915799])
    k2 = np.select([red_region, green_region, blue_region], [0.59662641, 0.08285427, -1.15130210])
    k3 = np.select([red_region, green_region, blue_region], [0.75515197, 0.12541070, -0.50559606])
    k4 = np.select([red_region, green_region, blue_region], [0.56771245, 0.14503204, 0.00692167])
    weight_l = np.select([red_region, green_region, blue_region], [4.0767416621, -1.2684380046, -0.0041960863])
    weight_m = np.select([red_region, green_region, blue_region], [-3.3077115913, 2.6097574011, -0.7034186147])
    weight_s = np.select([red_region, green_region, blue_region], [0.2309699292, -0.3413193965, 1.7076147010])

    saturation = k0 + k1 * a_unit + k2 * b_unit + k3 * a_unit * a_unit + k4 * a_unit * b_unit

    k_l = 0.3963377774 * a_unit + 0.2158037573 * b_unit
    k_m = -0.1055613458 * a_unit - 0.0638541728 * b_unit
    k_s = -0.0894841775 * a_unit - 1.2914855480 * b_unit

    l_root = 1.0 + saturation * k_l
    m_root = 1.0 + saturation * k_m
    s_root = 1.0 + saturation * k_s

    l_channel = l_root * l_root * l_root
    m_channel = m_root * m_root * m_root
    s_channel = s_root * s_root * s_root

    l_d1 = 3.0 * k_l * l_root * l_root
    m_d1 = 3.0 * k_m * m_root * m_root
    s_d1 = 3.0 * k_s * s_root * s_root

    l_d2 = 6.0 * k_l * k_l * l_root
    m_d2 = 6.0 * k_m * k_m * m_root
    s_d2 = 6.0 * k_s * k_s * s_root

    f_value = weight_l * l_channel + weight_m * m_channel + weight_s * s_channel
    f_d1 = weight_l * l_d1 + weight_m * m_d1 + weight_s * s_d1
    f_d2 = weight_l * l_d2 + weight_m * m_d2 + weight_s * s_d2
    return saturation - f_value * f_d1 / (f_d1 * f_d1 - 0.5 * f_value * f_d2)


def find_oklab_cusp(a_unit: np.ndarray, b_unit: np.ndarray):
    """求给定 hue 方向在 sRGB gamut 边界上的 cusp。"""
    saturation_cusp = compute_oklab_max_saturation(a_unit, b_unit)
    cusp_lab = np.stack([np.ones_like(a_unit), saturation_cusp * a_unit, saturation_cusp * b_unit], axis=-1)
    rgb_at_cusp = oklab_to_linear_srgb(cusp_lab)
    rgb_max = np.maximum(np.max(rgb_at_cusp, axis=-1), 1e-8)
    lightness_cusp = np.cbrt(1.0 / rgb_max)
    chroma_cusp = lightness_cusp * saturation_cusp
    return lightness_cusp, chroma_cusp


def find_oklab_gamut_intersection_preserve_lightness(
    lightness: np.ndarray,
    chroma: np.ndarray,
    a_unit: np.ndarray,
    b_unit: np.ndarray,
) -> np.ndarray:
    """在固定 L 和 hue 的前提下，求 Oklch 射线与 sRGB gamut 的解析交点比例。"""
    lightness_cusp, chroma_cusp = find_oklab_cusp(a_unit, b_unit)
    denominator = chroma * lightness_cusp
    return np.clip(
        np.divide(
            chroma_cusp * lightness,
            denominator,
            out=np.zeros_like(chroma),
            where=np.abs(denominator) > 1e-12,
        ),
        0.0,
        1.0,
    )


def _collapse_alpha_mask_to_scalar(mask_float: np.ndarray) -> np.ndarray:
    """Collapse a mask image with arbitrary channels into one scalar alpha plane."""
    mask_float = np.asarray(mask_float, dtype=np.float64)
    if mask_float.ndim == 2:
        return mask_float
    if mask_float.ndim != 3 or mask_float.shape[2] < 1:
        raise ValueError("alpha mask image must be grayscale or have at least one channel")

    channel_count = mask_float.shape[2]
    if channel_count == 1:
        return mask_float[:, :, 0]

    if channel_count in {2, 4}:
        explicit_alpha = mask_float[:, :, -1]
        if not np.allclose(explicit_alpha, 1.0, atol=ALPHA_VALID_EPSILON):
            return explicit_alpha
        color_channels = mask_float[:, :, :-1]
    else:
        color_channels = mask_float[:, :, : min(channel_count, 3)]

    if color_channels.shape[2] == 1:
        return color_channels[:, :, 0]
    if color_channels.shape[2] >= 3:
        return np.tensordot(
            color_channels[:, :, :3],
            np.array([0.2126, 0.7152, 0.0722], dtype=np.float64),
            axes=([-1], [0]),
        )
    return np.mean(color_channels, axis=-1)


def _load_alpha_mask(alpha_mask_path: str, expected_shape: tuple[int, int]) -> np.ndarray:
    """Load and validate an external alpha mask image."""
    mask_image = io.imread(alpha_mask_path)
    mask_float = img_as_float(mask_image)
    alpha_float = np.clip(_collapse_alpha_mask_to_scalar(mask_float), 0.0, 1.0)
    if alpha_float.shape != expected_shape:
        raise ValueError(
            "alpha mask image must match the source image size exactly: "
            f"expected {expected_shape}, got {alpha_float.shape}"
        )
    return np.asarray(alpha_float, dtype=np.float64)


def _build_border_mask(image_shape: tuple[int, int], border_width: int) -> np.ndarray:
    """Build a boolean mask that covers the outer border strip of an image."""
    height, width = image_shape
    effective_width = max(1, min(int(border_width), max(1, min(height, width) // 2)))
    border_mask = np.zeros((height, width), dtype=bool)
    border_mask[:effective_width, :] = True
    border_mask[-effective_width:, :] = True
    border_mask[:, :effective_width] = True
    border_mask[:, -effective_width:] = True
    return border_mask


def _longest_true_run(values: np.ndarray) -> int:
    """Return the longest contiguous run of True values in a 1D boolean array."""
    if values.size == 0:
        return 0
    padded = np.concatenate([
        np.array([False], dtype=bool),
        np.asarray(values, dtype=bool),
        np.array([False], dtype=bool),
    ])
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    if starts.size == 0:
        return 0
    return int(np.max(ends - starts))


def _candidate_has_edge_run(candidate_mask: np.ndarray, border_width: int, min_edge_run: int) -> bool:
    """Check whether a candidate border mask contains a sufficiently long edge run."""
    height, width = candidate_mask.shape
    effective_width = max(1, min(int(border_width), max(1, min(height, width) // 2)))
    edge_lines = []
    edge_lines.extend(candidate_mask[row, :] for row in range(effective_width))
    edge_lines.extend(candidate_mask[height - 1 - row, :] for row in range(effective_width))
    edge_lines.extend(candidate_mask[:, column] for column in range(effective_width))
    edge_lines.extend(candidate_mask[:, width - 1 - column] for column in range(effective_width))
    return any(_longest_true_run(line) >= int(min_edge_run) for line in edge_lines)


def detect_auto_valid_mask(
    rgb_float: np.ndarray,
    *,
    border_width: int = AUTO_MASK_BORDER_WIDTH,
    color_tolerance: int = AUTO_MASK_COLOR_TOLERANCE,
    histogram_bin_size: int = AUTO_MASK_HISTOGRAM_BIN_SIZE,
    max_border_std: float = AUTO_MASK_MAX_BORDER_STD,
    min_edge_run: int = AUTO_MASK_MIN_EDGE_RUN,
    min_border_pixels: int = AUTO_MASK_MIN_BORDER_PIXELS,
) -> np.ndarray:
    """Auto-detect a border-connected invalid region and return a valid-pixel mask."""
    rgb_uint8 = img_as_ubyte(np.clip(np.asarray(rgb_float, dtype=np.float64), 0.0, 1.0))
    border_mask = _build_border_mask(rgb_uint8.shape[:2], border_width)
    border_pixels = rgb_uint8[border_mask].reshape(-1, 3)
    if border_pixels.size == 0:
        raise ValueError("auto-detect mask requires a non-empty image")

    binned_border_pixels = border_pixels // max(1, int(histogram_bin_size))
    unique_bins, counts = np.unique(binned_border_pixels, axis=0, return_counts=True)
    dominant_index = int(np.argmax(counts))
    dominant_bin = unique_bins[dominant_index]
    dominant_pixels = border_pixels[np.all(binned_border_pixels == dominant_bin, axis=1)]
    if dominant_pixels.shape[0] < max(int(min_border_pixels), int(min_edge_run)):
        raise ValueError("auto-detect mask could not find a stable border color candidate")

    dominant_std = np.std(dominant_pixels.astype(np.float64), axis=0)
    if float(np.max(dominant_std)) > float(max_border_std):
        raise ValueError("auto-detect mask found a border color candidate, but it varies too much to trust")

    dominant_color = np.round(np.mean(dominant_pixels.astype(np.float64), axis=0)).astype(np.int16)
    rgb_int16 = rgb_uint8.astype(np.int16)
    candidate_mask = np.max(np.abs(rgb_int16 - dominant_color[None, None, :]), axis=-1) <= int(color_tolerance)
    border_seed = candidate_mask & border_mask
    if np.count_nonzero(border_seed) < max(int(min_border_pixels), int(min_edge_run)):
        raise ValueError("auto-detect mask could not find enough border-connected pixels for the dominant color")
    if not _candidate_has_edge_run(border_seed, border_width, min_edge_run):
        raise ValueError("auto-detect mask could not find a long enough constant-color run on the image edge")

    invalid_mask = binary_propagation(border_seed, mask=candidate_mask)
    valid_mask = ~invalid_mask
    if not np.any(invalid_mask):
        raise ValueError("auto-detect mask did not find any border-connected invalid region")
    if not np.any(valid_mask):
        raise ValueError("auto-detect mask would remove the entire image")
    return np.asarray(valid_mask, dtype=bool)


def _build_image_alpha_warnings(
    image_path: str,
    has_embedded_alpha: bool,
    embedded_alpha: np.ndarray,
    alpha_source: str,
    mask_prompt_required: bool,
) -> tuple[str, ...]:
    """Build user-facing warnings related to alpha-channel handling."""
    warnings: list[str] = []
    suffix = Path(image_path).suffix.lower()
    opaque_png_alpha = suffix == ".png" and has_embedded_alpha and np.all(embedded_alpha >= 1.0 - ALPHA_VALID_EPSILON)

    if alpha_source == "external-mask":
        if suffix in JPEG_IMAGE_SUFFIXES:
            warnings.append("Input JPEG image has no embedded alpha channel; using the external alpha mask instead.")
        elif opaque_png_alpha:
            warnings.append("Input PNG alpha channel is fully opaque; using the external alpha mask instead.")
    elif alpha_source == "auto-detected":
        if suffix in JPEG_IMAGE_SUFFIXES:
            warnings.append("Input JPEG image has no embedded alpha channel; using the auto-detected border mask instead.")
        elif opaque_png_alpha:
            warnings.append("Input PNG alpha channel is fully opaque; using the auto-detected border mask instead.")
        elif not has_embedded_alpha:
            warnings.append("Input image has no embedded alpha channel; using the auto-detected border mask instead.")
    elif alpha_source == "implicit-opaque":
        if suffix in JPEG_IMAGE_SUFFIXES:
            warnings.append("Input JPEG image has no embedded alpha channel.")
        elif opaque_png_alpha:
            warnings.append("Input PNG alpha channel is fully opaque and behaves like no mask.")
        elif not has_embedded_alpha:
            warnings.append("Input image has no embedded alpha channel.")
        if mask_prompt_required:
            warnings.append("No usable alpha mask was found. You can provide one or try auto-detect.")

    return tuple(warnings)


def load_image_data(
    image_path: str,
    *,
    alpha_mask_path: str | None = None,
    auto_detect_mask: bool = False,
) -> LoadedImageData:
    """Load RGB and alpha coverage data, preserving warnings and alpha provenance."""
    resolved_image_path = resolve_input_image_path(image_path)
    resolved_alpha_mask_path = resolve_optional_path(alpha_mask_path, argument_name="alpha mask image")

    original_image = io.imread(resolved_image_path)
    image_float = img_as_float(original_image)
    if image_float.ndim != 3 or image_float.shape[2] < 3:
        raise ValueError("input image must contain at least three color channels")

    rgb_float = np.asarray(image_float[:, :, :3], dtype=np.float64)
    has_embedded_alpha = image_float.shape[2] >= 4
    embedded_alpha = (
        np.asarray(image_float[:, :, 3], dtype=np.float64)
        if has_embedded_alpha
        else np.ones(rgb_float.shape[:2], dtype=np.float64)
    )
    embedded_alpha_is_usable = has_embedded_alpha and not np.all(embedded_alpha >= 1.0 - ALPHA_VALID_EPSILON)
    mask_prompt_required = False

    if resolved_alpha_mask_path is not None:
        alpha_float = _load_alpha_mask(resolved_alpha_mask_path, rgb_float.shape[:2])
        alpha_source = "external-mask"
    elif embedded_alpha_is_usable:
        alpha_float = embedded_alpha
        alpha_source = "embedded"
    elif auto_detect_mask:
        alpha_float = detect_auto_valid_mask(rgb_float).astype(np.float64)
        alpha_source = "auto-detected"
    else:
        alpha_float = np.ones(rgb_float.shape[:2], dtype=np.float64)
        alpha_source = "implicit-opaque"
        mask_prompt_required = True

    valid_mask = np.asarray(alpha_float > ALPHA_VALID_EPSILON, dtype=bool)
    oklch_float = rgb_to_oklch(rgb_float)

    return LoadedImageData(
        image_path=resolved_image_path,
        alpha_mask_path=resolved_alpha_mask_path,
        rgb_float=rgb_float,
        alpha_float=alpha_float,
        oklch_float=oklch_float,
        valid_mask=valid_mask,
        alpha_source=alpha_source,
        image_warnings=_build_image_alpha_warnings(
            resolved_image_path,
            has_embedded_alpha,
            embedded_alpha,
            alpha_source,
            mask_prompt_required,
        ),
        mask_prompt_required=mask_prompt_required,
    )


def load_image(
    image_path: str,
    *,
    alpha_mask_path: str | None = None,
    auto_detect_mask: bool = False,
):
    """Load an image and return the legacy RGB / Oklch / valid-mask triple."""
    loaded_image = load_image_data(
        image_path,
        alpha_mask_path=alpha_mask_path,
        auto_detect_mask=auto_detect_mask,
    )
    return loaded_image.rgb_float, loaded_image.oklch_float, loaded_image.valid_mask


def build_luma_preview_frame(
    rgb_float: np.ndarray,
    oklch_float: np.ndarray,
    valid_mask: np.ndarray,
    preview_scale: float = DEFAULT_FAST_PREVIEW_SCALE,
) -> LumaPreviewFrame:
    """按固定比例切片降采样，构建与 GUI 预览一致的快速输入帧。"""
    if preview_scale <= 0.0:
        raise ValueError("preview_scale must be > 0")

    height, width = oklch_float.shape[:2]
    preview_height = max(1, int(height * preview_scale))
    preview_width = max(1, int(width * preview_scale))
    step_y = max(1, height // preview_height)
    step_x = max(1, width // preview_width)

    rgb_small = rgb_float[::step_y, ::step_x]
    oklch_small = oklch_float[::step_y, ::step_x]
    mask_small = valid_mask[::step_y, ::step_x]
    return LumaPreviewFrame(
        rgb_float=rgb_small,
        oklch_float=oklch_small,
        valid_mask=mask_small,
        y_image=oklch_small[:, :, 0],
        scale=float(preview_scale),
    )


def compute_luma_lut_indices(y_image: np.ndarray, lut_size: int) -> np.ndarray:
    """将输入轴 y 量化到 LUT 采样索引。"""
    if lut_size < 2:
        raise ValueError("lut_size must be at least 2")
    return np.clip(np.round(y_image * (lut_size - 1)), 0, lut_size - 1).astype(np.int32)


def build_luma_preview_lut(
    state_curves: StateCurveSet,
    preview_lut_size: int = DEFAULT_FAST_LUT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """生成与 GUI 快速预览一致的 LUT，并返回被压缩的 LUT 条目标记。"""
    preview_x = np.linspace(0.0, 1.0, preview_lut_size)
    lightness = np.clip(state_curves.lightness_interp(preview_x), 0.0, 1.0)
    chroma = np.clip(state_curves.chroma_interp(lightness), 0.0, None)

    hue_u = state_curves.hue_u_interp(lightness)
    hue_v = state_curves.hue_v_interp(lightness)
    hue_norm = np.hypot(hue_u, hue_v)
    safe_norm = np.where(hue_norm < 1e-8, 1.0, hue_norm)
    hue = (np.degrees(np.arctan2(hue_v / safe_norm, hue_u / safe_norm)) + 360.0) % 360.0

    preview_oklch = np.column_stack([lightness, chroma, hue])
    adjusted_oklch, preview_rgb, _ = compress_oklch_chroma_to_srgb(preview_oklch)
    compressed_entries = np.abs(adjusted_oklch[:, 1] - preview_oklch[:, 1]) > 1e-10
    preview_lut_uint8 = np.clip(np.round(preview_rgb * 255.0), 0, 255).astype(np.uint8)
    return preview_lut_uint8, compressed_entries


def apply_luma_preview_lut(
    y_index_image: np.ndarray,
    valid_mask: np.ndarray | None,
    lut_uint8: np.ndarray,
    out_buf: np.ndarray | None = None,
) -> np.ndarray:
    """在量化索引图上做 LUT 查表，返回 uint8 RGB 图。"""
    if out_buf is None:
        out_buf = np.empty((*y_index_image.shape, 3), dtype=np.uint8)
    np.take(lut_uint8, y_index_image, axis=0, out=out_buf)
    if valid_mask is not None:
        out_buf[~valid_mask] = 0
    return out_buf


def count_luma_preview_gamut_pixels(
    y_index_image: np.ndarray,
    valid_mask: np.ndarray,
    compressed_entries: np.ndarray,
) -> int:
    """统计预览图中实际引用到被 gamut compression 的 LUT 条目的像素数。"""
    return int(np.count_nonzero(valid_mask & compressed_entries[y_index_image]))


def quantize_rgb_image(rgb_float: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """把浮点 RGB 图量化成 uint8，并对透明区域清零。"""
    rgb_uint8 = img_as_ubyte(np.clip(rgb_float, 0.0, 1.0))
    if valid_mask is not None:
        rgb_uint8[~valid_mask] = 0
    return rgb_uint8


def save_luma_output_image(output_image: np.ndarray, output_path: str):
    """保存 CLI / GUI 共用的结果图像。"""
    io.imsave(output_path, output_image)


def normalize_luma_execution_request(request: LumaExecutionRequest) -> LumaExecutionRequest:
    """补齐默认值并校验统一执行请求。"""
    normalized = LumaExecutionRequest(
        image_path=resolve_input_image_path(request.image_path),
        alpha_mask_path=resolve_optional_path(request.alpha_mask_path, argument_name="alpha mask image"),
        curve_path=request.curve_path,
        algorithm=request.algorithm or "original",
        dither_strength=float(request.dither_strength),
        evaluate_result=bool(request.evaluate_result),
        show_plots=bool(request.show_plots),
        preview_scale=float(request.preview_scale),
        preview_lut_size=int(request.preview_lut_size),
        output_image_path=request.output_image_path,
        result_json_path=request.result_json_path,
    )

    if normalized.algorithm not in SUPPORTED_LUMA_ALGORITHMS:
        raise ValueError(f"algorithm must be one of {SUPPORTED_LUMA_ALGORITHMS}")
    if normalized.preview_scale <= 0.0:
        raise ValueError("preview_scale must be > 0")
    if normalized.preview_lut_size < 2:
        raise ValueError("preview_lut_size must be at least 2")
    return normalized


def luma_request_from_payload(payload: dict) -> LumaExecutionRequest:
    """从 JSON 兼容对象构建统一执行请求。"""
    if not isinstance(payload, dict):
        raise ValueError("request payload must be a JSON object")

    algorithm = payload.get("algorithm", "original")
    evaluate_result = payload.get("evaluate_result")
    if evaluate_result is None:
        evaluate_result = algorithm == "original"

    return normalize_luma_execution_request(
        LumaExecutionRequest(
            image_path=payload.get("image_path"),
            alpha_mask_path=payload.get("alpha_mask_path"),
            curve_path=payload.get("curve_path"),
            algorithm=algorithm,
            dither_strength=payload.get("dither_strength", DITHER_STRENGTH),
            evaluate_result=evaluate_result,
            show_plots=payload.get("show_plots", True),
            preview_scale=payload.get("preview_scale", DEFAULT_FAST_PREVIEW_SCALE),
            preview_lut_size=payload.get("preview_lut_size", DEFAULT_FAST_LUT_SIZE),
            output_image_path=payload.get("output_image_path"),
            result_json_path=payload.get("result_json_path"),
        )
    )


def load_luma_request_json(request_json_path: str) -> LumaExecutionRequest:
    """从 JSON 文件加载统一执行请求。"""
    with open(request_json_path, "r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    return luma_request_from_payload(payload)


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


def _prepare_lightness_sample_array(
    lightness_values: np.ndarray,
    *,
    arg_name: str,
) -> np.ndarray:
    """Normalize a lightness sample array into a clipped 1D float64 vector."""
    values = np.asarray(lightness_values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{arg_name} must contain at least one value")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{arg_name} must contain only finite values")
    return np.clip(values, 0.0, 1.0)


def _collapse_lightness_transfer_points(control_points: np.ndarray) -> np.ndarray:
    """Collapse duplicate x positions while preserving a non-decreasing y mapping."""
    unique_x, inverse = np.unique(control_points[:, 0], return_inverse=True)
    unique_y = np.zeros(unique_x.shape[0], dtype=np.float64)
    counts = np.bincount(inverse)
    np.add.at(unique_y, inverse, control_points[:, 1])
    unique_y /= counts
    unique_y = np.maximum.accumulate(np.clip(unique_y, 0.0, 1.0))
    return np.column_stack([unique_x, unique_y])


def fit_monotonic_lightness_transfer_curve(
    source_lightness: np.ndarray,
    target_lightness: np.ndarray,
    *,
    quantile_count: int = MIN_KEYPOINTS,
) -> np.ndarray:
    """Fit monotonic Lt(y) control points by matching source and target empirical CDFs.

    The returned control points can be passed directly as `lightness_control_points`
    to `build_state_curve_set(...)` or serialized into the existing curve JSON format.
    """
    if quantile_count < 2:
        raise ValueError("quantile_count must be at least 2")

    source_values = _prepare_lightness_sample_array(
        source_lightness,
        arg_name="source_lightness",
    )
    target_values = _prepare_lightness_sample_array(
        target_lightness,
        arg_name="target_lightness",
    )

    quantiles = np.linspace(0.0, 1.0, int(quantile_count))
    source_quantiles = np.quantile(source_values, quantiles)
    target_quantiles = np.quantile(target_values, quantiles)
    control_points = np.column_stack([source_quantiles, target_quantiles])

    if control_points[0, 0] > 0.0:
        control_points = np.vstack([
            np.array([[0.0, target_quantiles[0]]], dtype=np.float64),
            control_points,
        ])
    if control_points[-1, 0] < 1.0:
        control_points = np.vstack([
            control_points,
            np.array([[1.0, target_quantiles[-1]]], dtype=np.float64),
        ])

    control_points = _collapse_lightness_transfer_points(control_points)
    if control_points.shape[0] < 2:
        control_points = np.array(
            [[0.0, target_quantiles[0]], [1.0, target_quantiles[-1]]],
            dtype=np.float64,
        )

    return prepare_control_points(
        control_points,
        control_points[:, 0],
        control_points[:, 1],
        clip_min=0.0,
        clip_max=1.0,
    )


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
    """固定 L 和 h，仅通过解析求交后压缩 C 将颜色拉回 sRGB gamut。"""
    del iterations
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
    chroma = invalid_oklch[:, 1]
    hue = invalid_oklch[:, 2]
    hue_rad = np.deg2rad(hue)
    a_unit = np.cos(hue_rad)
    b_unit = np.sin(hue_rad)
    chroma_scale = find_oklab_gamut_intersection_preserve_lightness(lightness, chroma, a_unit, b_unit)

    adjusted_oklch = np.array(oklch_values, copy=True)
    adjusted_invalid = adjusted_oklch[invalid_mask]
    adjusted_invalid[:, 1] = chroma_scale * chroma
    adjusted_oklch[invalid_mask] = adjusted_invalid

    adjusted_rgb = np.clip(oklch_to_rgb(adjusted_oklch), 0.0, 1.0)
    adjusted_rgb[~valid_mask] = 0.0
    return adjusted_oklch, adjusted_rgb, int(np.count_nonzero(invalid_mask))


def reconstruct_from_state_curves(
    oklch_float: np.ndarray,
    valid_mask: np.ndarray,
    state_curves: StateCurveSet,
    dither_strength: float = DITHER_STRENGTH,
    output_valid_mask: np.ndarray | None = None,
):
    """对抖动后的输入轴先求 Lt，再通过 Ct / ht 生成最终颜色状态。"""
    if output_valid_mask is None:
        output_valid_mask = valid_mask
    else:
        output_valid_mask = np.asarray(output_valid_mask, dtype=bool)
        if output_valid_mask.shape != valid_mask.shape:
            raise ValueError("output_valid_mask must match valid_mask shape")

    y_image = oklch_float[:, :, 0]
    y_eval = apply_precurve_dither(y_image, output_valid_mask, dither_strength)
    lightness_eval, chroma_eval, hue_eval = evaluate_state_curves(state_curves, y_eval)
    reconstructed_oklch = np.stack([lightness_eval, chroma_eval, hue_eval], axis=-1)
    if not np.all(output_valid_mask):
        reconstructed_oklch[~output_valid_mask] = 0.0
    adjusted_oklch, reconstructed_rgb, gamut_compressed_pixels = compress_oklch_chroma_to_srgb(
        reconstructed_oklch, output_valid_mask
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


def run_original_luma_algorithm(request: LumaExecutionRequest) -> LumaExecutionResult:
    """执行原始高质量离线主流程。"""
    loaded_image = load_image_data(request.image_path, alpha_mask_path=request.alpha_mask_path)
    rgb_float = loaded_image.rgb_float
    oklch_float = loaded_image.oklch_float
    valid_mask = loaded_image.valid_mask
    output_valid_mask = np.ones_like(valid_mask, dtype=bool)
    model, y_samples = build_oklch_curve_model(oklch_float, valid_mask)
    curve_overrides = load_state_curve_overrides(request.curve_path)
    state_curves = build_state_curve_set(model, **curve_overrides)
    recolored_rgb_float, _, y_eval, gamut_compressed_pixels = reconstruct_from_state_curves(
        oklch_float,
        valid_mask,
        state_curves,
        dither_strength=request.dither_strength,
        output_valid_mask=output_valid_mask,
    )
    recolored_rgb_int = quantize_rgb_image(recolored_rgb_float)

    psnr = None
    delta_e_image = None
    delta_e_stats = None
    if request.evaluate_result:
        recolored_rgb_int, psnr, delta_e_image, delta_e_stats = evaluate_reconstruction(
            rgb_float,
            recolored_rgb_float,
            valid_mask,
        )

    return LumaExecutionResult(
        algorithm="original",
        image_path=request.image_path,
        alpha_mask_path=request.alpha_mask_path,
        curve_path=request.curve_path,
        dither_strength=request.dither_strength,
        source_image_shape=tuple(int(value) for value in rgb_float.shape[:2]),
        output_image_shape=tuple(int(value) for value in rgb_float.shape[:2]),
        output_scale=1.0,
        preview_lut_size=None,
        rgb_float=rgb_float,
        oklch_float=oklch_float,
        valid_mask=valid_mask,
        model=model,
        y_samples=y_samples,
        state_curves=state_curves,
        recolored_rgb_float=recolored_rgb_float,
        y_eval=y_eval,
        gamut_compressed_pixels=gamut_compressed_pixels,
        alpha_source=loaded_image.alpha_source,
        image_warnings=loaded_image.image_warnings,
        recolored_rgb_int=recolored_rgb_int,
        psnr=psnr,
        delta_e_image=delta_e_image,
        delta_e_stats=delta_e_stats,
    )


def run_fast_luma_algorithm(request: LumaExecutionRequest) -> LumaExecutionResult:
    """执行与未来 GUI 预览一致的快速 LUT 算法。"""
    loaded_image = load_image_data(request.image_path, alpha_mask_path=request.alpha_mask_path)
    source_rgb_float = loaded_image.rgb_float
    source_oklch_float = loaded_image.oklch_float
    source_valid_mask = loaded_image.valid_mask
    model, y_samples = build_oklch_curve_model(source_oklch_float, source_valid_mask)
    curve_overrides = load_state_curve_overrides(request.curve_path)
    state_curves = build_state_curve_set(model, **curve_overrides)

    preview_frame = build_luma_preview_frame(
        source_rgb_float,
        source_oklch_float,
        source_valid_mask,
        preview_scale=request.preview_scale,
    )
    y_eval = apply_precurve_dither(
        preview_frame.y_image,
        np.ones_like(preview_frame.valid_mask, dtype=bool),
        request.dither_strength,
    )
    y_index = compute_luma_lut_indices(y_eval, request.preview_lut_size)
    preview_lut_uint8, compressed_entries = build_luma_preview_lut(
        state_curves,
        preview_lut_size=request.preview_lut_size,
    )
    recolored_rgb_int = apply_luma_preview_lut(y_index, None, preview_lut_uint8)
    recolored_rgb_float = img_as_float(recolored_rgb_int)
    gamut_compressed_pixels = count_luma_preview_gamut_pixels(
        y_index,
        preview_frame.valid_mask,
        compressed_entries,
    )

    psnr = None
    delta_e_image = None
    delta_e_stats = None
    if request.evaluate_result:
        recolored_rgb_int, psnr, delta_e_image, delta_e_stats = evaluate_reconstruction(
            preview_frame.rgb_float,
            recolored_rgb_float,
            preview_frame.valid_mask,
        )

    return LumaExecutionResult(
        algorithm="fast",
        image_path=request.image_path,
        alpha_mask_path=request.alpha_mask_path,
        curve_path=request.curve_path,
        dither_strength=request.dither_strength,
        source_image_shape=tuple(int(value) for value in source_rgb_float.shape[:2]),
        output_image_shape=tuple(int(value) for value in preview_frame.rgb_float.shape[:2]),
        output_scale=preview_frame.scale,
        preview_lut_size=request.preview_lut_size,
        rgb_float=preview_frame.rgb_float,
        oklch_float=preview_frame.oklch_float,
        valid_mask=preview_frame.valid_mask,
        model=model,
        y_samples=y_samples,
        state_curves=state_curves,
        recolored_rgb_float=recolored_rgb_float,
        y_eval=y_eval,
        gamut_compressed_pixels=gamut_compressed_pixels,
        alpha_source=loaded_image.alpha_source,
        image_warnings=loaded_image.image_warnings,
        recolored_rgb_int=recolored_rgb_int,
        psnr=psnr,
        delta_e_image=delta_e_image,
        delta_e_stats=delta_e_stats,
        gamut_compressed_lut_entries=int(np.count_nonzero(compressed_entries)),
    )


def run_luma_workflow(request: LumaExecutionRequest) -> LumaExecutionResult:
    """按统一请求执行原始或快速算法。"""
    request = normalize_luma_execution_request(request)
    if request.algorithm == "original":
        return run_original_luma_algorithm(request)
    if request.algorithm == "fast":
        return run_fast_luma_algorithm(request)
    raise ValueError(f"Unsupported algorithm: {request.algorithm}")


def run_luma_color_map(
    image_path: str | None = None,
    *,
    alpha_mask_path: str | None = None,
    curve_path: str | None = None,
    dither_strength: float = DITHER_STRENGTH,
    evaluate_result: bool = True,
    algorithm: str = "original",
    preview_scale: float = DEFAULT_FAST_PREVIEW_SCALE,
    preview_lut_size: int = DEFAULT_FAST_LUT_SIZE,
) -> LumaExecutionResult:
    """统一入口：运行原始高质量算法或共享的快速 LUT 算法。"""
    return run_luma_workflow(
        LumaExecutionRequest(
            image_path=image_path,
            alpha_mask_path=alpha_mask_path,
            curve_path=curve_path,
            algorithm=algorithm,
            dither_strength=dither_strength,
            evaluate_result=evaluate_result,
            preview_scale=preview_scale,
            preview_lut_size=preview_lut_size,
        )
    )


def summarize_luma_result(result: LumaExecutionResult) -> dict:
    """将主流程结果压缩为适合日志或 JSON 的摘要。"""
    summary = {
        "algorithm": result.algorithm,
        "image_path": result.image_path,
        "alpha_mask_path": result.alpha_mask_path,
        "alpha_source": result.alpha_source,
        "image_warnings": list(result.image_warnings),
        "curve_path": result.curve_path,
        "curve_source": result.curve_path or "default base-model controls",
        "dither_strength": float(result.dither_strength),
        "source_image_shape": list(result.source_image_shape),
        "output_image_shape": list(result.output_image_shape),
        "output_scale": float(result.output_scale),
        "preview_lut_size": result.preview_lut_size,
        "keypoints": int(result.model.key_y.size),
        "state_curve_points": {
            "lightness": int(result.state_curves.lightness_points.shape[0]),
            "chroma": int(result.state_curves.chroma_points.shape[0]),
            "hue": int(result.state_curves.hue_points.shape[0]),
        },
        "gamut_compressed_pixels": int(result.gamut_compressed_pixels),
        "evaluation_enabled": result.delta_e_stats is not None,
        "output_image_path": result.output_image_path,
    }
    if result.gamut_compressed_lut_entries is not None:
        summary["gamut_compressed_lut_entries"] = int(result.gamut_compressed_lut_entries)
    if result.psnr is not None:
        summary["psnr"] = float(result.psnr)
    if result.delta_e_stats is not None:
        summary["delta_e_stats"] = {
            key: float(value) for key, value in result.delta_e_stats.items()
        }
    return summary


def write_luma_summary_json(result: LumaExecutionResult, output_path: str):
    """将主流程摘要写入 JSON 文件。"""
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summarize_luma_result(result), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


__all__ = [
    "ALPHA_VALID_EPSILON",
    "AUTO_MASK_BORDER_WIDTH",
    "AUTO_MASK_COLOR_TOLERANCE",
    "DATA_DIRECTORY",
    "DEFAULT_FAST_LUT_SIZE",
    "DEFAULT_FAST_PREVIEW_SCALE",
    "DEFAULT_SAMPLE_IMAGE_CANDIDATES",
    "DITHER_STRENGTH",
    "HUE_CHROMA_FLOOR",
    "LoadedImageData",
    "LumaExecutionRequest",
    "LumaExecutionResult",
    "LumaPreviewFrame",
    "MIN_KEYPOINTS",
    "OklchCurveModel",
    "STATE_CURVE_CTRL_POINTS",
    "SUPPORTED_LUMA_ALGORITHMS",
    "StateCurveSet",
    "aggregate_keypoint_samples",
    "apply_luma_preview_lut",
    "apply_precurve_dither",
    "build_luma_preview_frame",
    "build_luma_preview_lut",
    "build_oklch_curve_model",
    "build_state_curve_set",
    "compress_oklch_chroma_to_srgb",
    "compute_luma_lut_indices",
    "compute_oklab_max_saturation",
    "count_luma_preview_gamut_pixels",
    "detect_auto_valid_mask",
    "evaluate_chroma_hue",
    "evaluate_reconstruction",
    "evaluate_state_curves",
    "extract_quantile_keypoints",
    "find_oklab_cusp",
    "find_oklab_gamut_intersection_preserve_lightness",
    "fit_monotonic_lightness_transfer_curve",
    "linear_srgb_to_oklab",
    "linear_to_srgb",
    "load_image",
    "load_image_data",
    "load_luma_request_json",
    "load_state_curve_overrides",
    "luma_request_from_payload",
    "normalize_luma_execution_request",
    "oklab_to_linear_srgb",
    "oklch_to_rgb",
    "prepare_control_points",
    "quantize_rgb_image",
    "reconstruct_from_state_curves",
    "resolve_input_image_path",
    "rgb_to_oklch",
    "run_fast_luma_algorithm",
    "run_luma_color_map",
    "run_luma_workflow",
    "run_original_luma_algorithm",
    "save_luma_output_image",
    "srgb_to_linear",
    "summarize_luma_result",
    "write_luma_summary_json",
]