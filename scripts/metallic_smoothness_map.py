import numpy as np
from PIL import Image


def hex_to_rgba(hex_color: str) -> tuple:
    """将 #RRGGBBAA 格式的字符串转换为 (R, G, B, A) 元组。"""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4, 6))


def generate_maps(
    input_image_path: str,
    ms_output_path: str,
    color_output_path: str,
    materials: list[dict],
    threshold: int = 127,
):
    """
    从带蒙版的 RGBA 输入图像生成金属度/光滑度贴图和颜色贴图。

    Parameters
    ----------
    input_image_path : str
        输入蒙版图片路径（需含 Alpha 通道）。
    ms_output_path : str
        金属度/光滑度贴图输出路径。
    color_output_path : str
        颜色贴图输出路径。
    materials : list[dict]
        每个元素对应一个 RGB 通道的材质定义，包含：
        - channel: int (0=R, 1=G, 2=B)
        - metallic: float (0.0-1.0)
        - smoothness: float (0.0-1.0)
        - color_hex: str (#RRGGBBAA)
    threshold : int
        通道判定阈值 (0-255)。
    """
    input_image = Image.open(input_image_path).convert("RGBA")
    input_array = np.array(input_image)

    if input_array.shape[2] != 4:
        raise ValueError("输入图片不包含 Alpha 通道。")

    h, w = input_array.shape[:2]
    ms_output_array = np.zeros((h, w, 4), dtype=np.uint8)
    color_output_array = np.zeros((h, w, 4), dtype=np.uint8)

    for mat in materials:
        ch = mat["channel"]
        mask = input_array[:, :, ch] > threshold

        m_val = int(mat["metallic"] * 255)
        s_val = int(mat["smoothness"] * 255)
        rgba = hex_to_rgba(mat["color_hex"])

        ms_output_array[mask, 0] = m_val
        ms_output_array[mask, 1] = s_val
        color_output_array[mask] = rgba

    # 复制源 Alpha 通道
    source_alpha = input_array[:, :, 3]
    ms_output_array[:, :, 3] = source_alpha
    color_output_array[:, :, 3] = source_alpha

    Image.fromarray(ms_output_array).save(ms_output_path)
    print(f"已保存金属度/光滑度贴图: {ms_output_path}")

    Image.fromarray(color_output_array).save(color_output_path)
    print(f"已保存颜色贴图: {color_output_path}")


if __name__ == "__main__":
    materials = [
        {  # R通道 - 金缮
            "channel": 0,
            "metallic": 1.0,
            "smoothness": 0.445,
            "color_hex": "#D69687FF",
        },
        {  # G通道 - 陶瓷
            "channel": 1,
            "metallic": 0.0,
            "smoothness": 1.0,
            "color_hex": "#FBFFFCFF",
        },
        {  # B通道 - 黑锆金
            "channel": 2,
            "metallic": 0.0,
            "smoothness": 0.727,
            "color_hex": "#393938FF",
        },
    ]

    generate_maps(
        input_image_path="/content/MobiusBracelet-color.png",
        ms_output_path="metallic_smoothness_map.png",
        color_output_path="color_map.png",
        materials=materials,
        threshold=127,
    )
