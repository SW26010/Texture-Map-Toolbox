import numpy as np
from skimage import io, color
from skimage.util import img_as_float, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt
import colour


def load_image(image_path: str):
    """加载图像，提取 RGB 和 Alpha 通道（均为归一化浮点），计算灰度图。"""
    original_image = io.imread(image_path)
    image_float = img_as_float(original_image)
    rgb_float = image_float[:, :, :3]
    alpha_float = (
        image_float[:, :, 3]
        if original_image.shape[2] == 4
        else np.ones(rgb_float.shape[:2], dtype=rgb_float.dtype)
    )
    valid_mask = alpha_float > 0.5
    luma_float = color.rgb2gray(rgb_float)
    return rgb_float, luma_float, valid_mask


def _luma_to_uint8(luma_float):
    """将归一化亮度 [0,1] 量化为 uint8 [0,255]。"""
    return np.clip(np.round(luma_float * 255), 0, 255).astype(np.uint8)


def build_luma_color_lut(luma_float, rgb_float, valid_mask):
    """
    为每个灰度值 (0-255) 计算原图中对应像素的平均 RGB 颜色，
    返回 256x3 的 RGB 查找表和 256x3 的 HSL 查找表。

    内部计算全部使用归一化浮点，仅在量化灰度索引时转为 uint8。
    """
    luma_uint8 = _luma_to_uint8(luma_float)
    valid_luma_pixel_int = luma_uint8[valid_mask]
    valid_rgb_pixel_float = rgb_float[valid_mask]

    luma_histogram = np.bincount(valid_luma_pixel_int, minlength=256)

    rgb_sums_by_luma = np.stack(
        [
            np.bincount(valid_luma_pixel_int, weights=ch, minlength=256)
            for ch in valid_rgb_pixel_float.T
        ],
        axis=1,
    )

    # 安全除法：未出现的灰度值结果为 0 而非 NaN
    safe_counts = np.maximum(luma_histogram, 1)[:, np.newaxis]
    luma_to_rgb_map = rgb_sums_by_luma / safe_counts
    # 标记未出现的灰度值为 NaN（仅用于可视化分析）
    absent_mask = luma_histogram == 0
    luma_to_rgb_map[absent_mask] = np.nan

    luma_to_hsl_map = colour.RGB_to_HSL(luma_to_rgb_map)
    luma_to_hsl_map[absent_mask] = np.nan

    return luma_to_rgb_map, luma_to_hsl_map, luma_histogram, luma_uint8


def recolor_and_evaluate(rgb_float, luma_uint8, valid_mask, luma_to_rgb_map):
    """
    用 LUT 重新上色并计算 PSNR 和 Delta E 2000。

    所有计算在归一化浮点空间完成，仅最终输出时量化为 uint8。
    Delta E 仅对有效像素计算以避免浪费。
    """
    # LUT 查表（浮点空间），NaN 条目用 0 替代以避免 ubyte 转换出错
    lut_safe = np.nan_to_num(luma_to_rgb_map, nan=0.0)
    recolored_rgb_float = lut_safe[luma_uint8]

    # 量化到 8-bit 并转回浮点，模拟实际输出精度损失
    recolored_rgb_int = img_as_ubyte(np.clip(recolored_rgb_float, 0.0, 1.0))
    recolored_quantized_float = img_as_float(recolored_rgb_int)

    # PSNR：比较 uint8 空间（与实际使用场景一致）
    psnr = peak_signal_noise_ratio(
        img_as_ubyte(rgb_float)[valid_mask], recolored_rgb_int[valid_mask]
    )

    # Delta E 2000：仅对有效像素计算，节省色彩空间转换开销
    valid_recolored = recolored_quantized_float[valid_mask]
    valid_original = rgb_float[valid_mask]

    delta_e_valid = colour.difference.delta_E_CIE2000(
        colour.XYZ_to_Lab(colour.sRGB_to_XYZ(valid_recolored)),
        colour.XYZ_to_Lab(colour.sRGB_to_XYZ(valid_original)),
    )

    # 重建全图 Delta E 用于可视化
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


def plot_comparison(rgb_float, luma_float, recolored_rgb_int, valid_mask, psnr, delta_e_image):
    """绘制灰度图、原图、重建图、色差误差图四宫格对比。"""
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    ax[0].imshow(np.where(valid_mask, luma_float, 0.0), cmap="gray")
    ax[0].set_title("Luma Image")

    ax[1].imshow(np.where(valid_mask[..., np.newaxis], img_as_ubyte(rgb_float), 0))
    ax[1].set_title("Original Image")

    ax[2].imshow(np.where(valid_mask[..., np.newaxis], recolored_rgb_int, 0))
    ax[2].set_title(f"Re-colored from Luma\nPSNR = {psnr:.2f} dB")

    DELTA_E_MAX = 1.0
    delta_e_clipped = np.clip(delta_e_image, 0.0, DELTA_E_MAX)
    im_de = ax[3].imshow(
        np.where(valid_mask, delta_e_clipped, 0.0), cmap="gray", vmin=0.0, vmax=DELTA_E_MAX
    )
    ax[3].set_title(f"CIEDE2000 Error Map (0-{DELTA_E_MAX})")
    pos3 = ax[3].get_position()
    cax = fig.add_axes([pos3.x1 + 0.005, pos3.y0, 0.015, pos3.height])
    cbar = fig.colorbar(im_de, cax=cax)
    cbar.set_label("Delta E 2000 Value")

    for a in ax:
        a.axis("off")
    plt.show(block=False)


def plot_analysis(luma_histogram, luma_to_rgb_map, luma_to_hsl_map):
    """绘制亮度直方图、L*/色相/饱和度曲线、LUT 可视化等分析图。"""
    fig, ax = plt.subplots(5, 1, figsize=(15, 13), sharex=True)

    # 1. 亮度直方图
    ax[0].bar(np.arange(256), luma_histogram, width=1.0, color="royalblue", alpha=0.8)
    ax[0].set_yscale("log")
    y_min, y_max = ax[0].get_ylim()
    ax[0].imshow(
        np.linspace(0, 1, 256).reshape(1, 256),
        cmap="gray", aspect="auto", extent=[0, 255, y_min, y_max],
    )
    ax[0].set_title("Luminance Histogram of Valid Pixels")
    ax[0].grid(axis="y", linestyle="--", alpha=0.6)
    ax[0].set_xlim(0, 255)

    # 2. Lightness 曲线
    l_channel = luma_to_hsl_map[:, 2]
    ax[1].plot(np.arange(256), l_channel, color="red", linewidth=2)
    ax[1].set_ylim(0, 1)
    y_min, y_max = ax[1].get_ylim()
    ax[1].imshow(l_channel.reshape(1, 256), cmap="gray", aspect="auto", extent=[0, 255, y_min, y_max])
    ax[1].grid(axis="y", linestyle="--", alpha=0.6)
    ax[1].set_title("Resulting Lightness (L*)")

    # 3. LUT 可视化
    ax[2].imshow(np.broadcast_to(luma_to_rgb_map, (32, 256, 3)))
    ax[2].set_title("Luminance to Average Color LUT")
    ax[2].set_yticks([])

    # 4. Hue 色度图
    h_channel = luma_to_hsl_map[:, 0] * 360
    ax[3].plot(np.arange(256), h_channel, color="black", linewidth=1.5, alpha=0.8)
    ymin, ymax = ax[3].get_ylim()

    height, width = 360, 255
    hue_gradient_hsl = np.zeros((height, width, 3), dtype=np.float64)
    hue_gradient_hsl[..., 0] = np.linspace(ymin / 360, ymax / 360, height)[:, np.newaxis]
    hue_gradient_hsl[..., 1] = 1.0
    hue_gradient_hsl[..., 2] = 0.5
    hue_gradient_rgb = colour.HSL_to_RGB(hue_gradient_hsl)
    ax[3].imshow(hue_gradient_rgb, origin="lower", aspect="auto", extent=[0, 255, ymin, ymax])
    ax[3].set_title("Hue Channel")

    # 5. Saturation 曲线
    s_channel = luma_to_hsl_map[:, 1]
    ax[4].plot(np.arange(256), s_channel, color="deepskyblue", linewidth=2)
    ax[4].set_title("Resulting Saturation (S)")
    ax[4].grid(axis="y", linestyle="--", alpha=0.6)
    ax[4].set_xlabel("Input Luminance (0-255)")

    plt.tight_layout()
    plt.show(block=False)


if __name__ == "__main__":
    import sys, os
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = os.path.join(os.path.dirname(__file__), "..", "data", "mtmtPonyTail.png")

    rgb_float, luma_float, valid_mask = load_image(image_path)
    luma_to_rgb_map, luma_to_hsl_map, luma_histogram, luma_uint8 = build_luma_color_lut(
        luma_float, rgb_float, valid_mask
    )
    recolored_rgb_int, psnr, delta_e_image, delta_e_stats = recolor_and_evaluate(
        rgb_float, luma_uint8, valid_mask, luma_to_rgb_map
    )

    print(f"PSNR (Peak Signal-to-Noise Ratio): {psnr:.2f} dB")
    for key, value in delta_e_stats.items():
        print(f"Delta E 2000 ({key.replace('_', ' ').title()}): {value:.2f}")

    plot_comparison(rgb_float, luma_float, recolored_rgb_int, valid_mask, psnr, delta_e_image)
    plot_analysis(luma_histogram, luma_to_rgb_map, luma_to_hsl_map)
    plt.show()  # 阻塞等待所有窗口关闭
