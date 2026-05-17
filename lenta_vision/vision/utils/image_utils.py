# common/image_utils.py
import cv2
import numpy as np
from PIL import Image

def bgr_to_pil_rgb(bgr_image: np.ndarray) -> Image.Image:
    """Конвертация cv2 (BGR) → PIL (RGB) только для LLM/визуализации"""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def pil_rgb_to_bgr(pil_image: Image.Image) -> np.ndarray:
    """Обратная конвертация, если нужно вернуть в cv2-пайплайн"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)