import cv2
import numpy as np

def get_blur_score(image: np.ndarray) -> float:
    """
    Рассчитывает оценку резкости изображения.
    Чем выше число, тем четче кадр.
    """
    # Переводим в ЧБ, так как резкость оценивается по яркостному каналу
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Считаем дисперсию Лапласиана
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score

def get_blur_score_robust(image: np.ndarray) -> float:
    """
    Рассчитывает оценку резкости, независимую от тусклого освещения.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Нормализуем яркость: самые темные пиксели станут 0, самые светлые 255.
    # Это искусственно вытянет контраст на темных кадрах перед проверкой резкости.
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Считаем дисперсию Лапласиана уже по нормализованной картинке
    score = cv2.Laplacian(normalized, cv2.CV_64F).var()
    return score

def is_not_blurred(image: np.ndarray, threshold: float) -> tuple[bool, float]:
    """Проверяет, достаточно ли резкий кадр."""
    # Используем нашу функцию с нормализацией яркости
    score = get_blur_score_robust(image)
    return score >= threshold, score