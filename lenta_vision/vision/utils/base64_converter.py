import cv2
import base64
import numpy as np

class ImageConverter:
    @staticmethod
    def cv2_to_base64(image: np.ndarray, ext: str = '.jpg') -> str:
        """
        Конвертирует изображение OpenCV (NumPy array BGR) в base64 строку.

        :param image: Изображение в формате cv2 (np.ndarray)
        :param ext: Формат кодирования (по умолчанию '.jpg')
        :return: Строка base64
        """
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Некорректный формат изображения. Ожидается numpy.ndarray.")

        # Кодируем изображение в буфер памяти
        success, buffer = cv2.imencode(ext, image)
        if not success:
            raise RuntimeError("Не удалось закодировать изображение в буфер.")

        # Конвертируем буфер в base64 и декодируем в строку
        b64_str = base64.b64encode(buffer).decode('utf-8')
        return b64_str

    @staticmethod
    def base64_to_cv2(b64_str: str) -> np.ndarray:
        """
        Опционально: Обратная конвертация (может пригодиться для дебага ответов сервера).
        """
        img_data = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img