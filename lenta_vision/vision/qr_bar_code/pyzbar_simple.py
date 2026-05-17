import cv2
from pyzbar.pyzbar import decode
import numpy as np

def read_barcode_basic(image: np.ndarray) -> list[dict]:
    """
    Простое распознавание штрихкодов.
    Переводит в ЧБ для лучшего считывания и ищет коды.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    decoded_objects = decode(gray)

    results = []
    for obj in decoded_objects:
        results.append({
            "data": obj.data.decode("utf-8"),
            "type": obj.type
        })

    return results

