"""Matplotlib presenters for the Oklch luma workflows."""

import matplotlib.pyplot as plt
import numpy as np
from skimage.util import img_as_ubyte

from texture_map_toolbox.core.luma import (
    OklchCurveModel,
    compress_oklch_chroma_to_srgb,
    evaluate_chroma_hue,
)


def plot_comparison(
    rgb_float: np.ndarray,
    y_image: np.ndarray,
    recolored_rgb_int: np.ndarray,
    valid_mask: np.ndarray,
    psnr: float,
    delta_e_image: np.ndarray,
):
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

    return fig


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
    return fig


__all__ = ["plot_analysis", "plot_comparison"]