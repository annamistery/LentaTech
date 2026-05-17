import cv2
from cv2 import dnn_superres
import numpy as np

def simple_ai_upscale(image: np.ndarray, model_path: str = "EDSR_x3.pb") -> np.ndarray:
    """
    Простой AI-апскейл через встроенный модуль OpenCV.
    """
    # 1. Создаем объект апскейлера
    sr = dnn_superres.DnnSuperResImpl_create()

    # 2. Читаем файл весов
    try:
        sr.readModel(model_path)
    except Exception as e:
        print(f"Ошибка загрузки модели. Проверьте, что файл {model_path} скачан и лежит рядом.")
        raise e

    # 3. Указываем название алгоритма и масштаб (x3)
    # Название алгоритма должно быть в нижнем регистре: "edsr", "fsrcnn", "lapsrn" и т.д.
    sr.setModel("edsr", 3)

    # 4. Делаем апскейл
    upscaled = sr.upsample(image)

    return upscaled

# --- Пример использования ---
if __name__ == "__main__":
    img = cv2.imread("dataset/crops/test_crop.jpg")

    if img is not None:
        print(f"Размер до: {img.shape}")

        # Запускаем функцию
        result = simple_ai_upscale(img, "EDSR_x3.pb")

        print(f"Размер после: {result.shape}")
        cv2.imwrite("dataset/crops/upscaled_edsr.jpg", result)