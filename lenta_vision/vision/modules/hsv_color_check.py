import cv2
import numpy as np

def detect_price_tag_color(image: np.ndarray, threshold: float = 0.25) -> str:
    """
    Определяет тип ценника (red, yellow, white, other).
    Ищет конкретно белый цвет, чтобы отсеять мусорные кропы (бутылки, полки).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    total_pixels = image.shape[0] * image.shape[1]

    # 1. ЖЕЛТЫЙ
    lower_yellow = np.array([20, 70, 70])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 2. КРАСНЫЙ / ОРАНЖЕВЫЙ
    lower_red1 = np.array([0, 70, 70])
    upper_red1 = np.array([15, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, 70, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # 3. БЕЛЫЙ (Новый блок!)
    # H: 0-180 (любой оттенок)
    # S: 0-40 (очень низкая насыщенность, допускаем легкие грязные тени)
    # V: 180-255 (высокая яркость, чтобы не спутать с черным штрихкодом или полкой)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # 4. Подсчет процентов
    yellow_percent = cv2.countNonZero(mask_yellow) / total_pixels
    red_percent = cv2.countNonZero(mask_red) / total_pixels
    white_percent = cv2.countNonZero(mask_white) / total_pixels

    # Можно раскомментировать для дебага:
    # print(f"Красный: {red_percent:.1%}, Желтый: {yellow_percent:.1%}, Белый: {white_percent:.1%}")

    # 5. Принимаем решение
    if red_percent >= threshold:
        return "red"
    elif yellow_percent >= threshold:
        return "yellow"
    elif white_percent >= threshold:
        return "white"
    else:
        # Если ни один цвет не набрал 25%, значит в кадре каша (бутылка, полка, темный фон)
        return "other"

# --- Пример использования ---
if __name__ == "__main__":
    # Ваш последний загруженный кроп
    img = cv2.imread("/vision/_img/yell.jpg")

    if img is not None:
        tag_color = detect_price_tag_color(img, threshold=0.25)
        print(f"Цвет ценника: {tag_color}")