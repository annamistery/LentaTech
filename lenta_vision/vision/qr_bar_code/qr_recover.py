import cv2
import numpy as np
from pyzbar.pyzbar import decode

def reconstruct_qr_hardcoded(image: np.ndarray, grid_size: int = 25, output_size: int = 500) -> np.ndarray:
    """
    Продвинутое восстановление QR-кода: авто-обрезка белых краев и
    жесткая отрисовка идеальных угловых маркеров.
    """
    # 0. Бинаризация
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 1. АВТО-ОБРЕЗКА БЕЛЫХ ПОЛЕЙ
    # Инвертируем картинку (черный фон, белые пиксели кода),
    # чтобы найти все значащие пиксели
    binary_inv = cv2.bitwise_not(binary)
    coords = cv2.findNonZero(binary_inv)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Вырезаем строго сам код, без полей
        cropped = binary[y:y+h, x:x+w]
    else:
        cropped = binary # Fallback, если картинка пустая

    height, width = cropped.shape
    cell_h = height / grid_size
    cell_w = width / grid_size

    # Создаем виртуальную матрицу кода (массив 25х25, где 255-белый, 0-черный)
    matrix = np.ones((grid_size, grid_size), dtype=np.uint8) * 255

    # 2. СЕМПЛИРОВАНИЕ (как в прошлом скрипте, но по обрезанной картинке)
    for row in range(grid_size):
        for col in range(grid_size):
            y1, y2 = int(row * cell_h), int((row + 1) * cell_h)
            x1, x2 = int(col * cell_w), int((col + 1) * cell_w)

            patch = cropped[y1:y2, x1:x2]
            if np.mean(patch) < 128:
                matrix[row, col] = 0

    # 3. ХАРДКОД УГЛОВЫХ МАРКЕРОВ (Finder Patterns)
    # Они всегда 7х7. Мы просто переписываем углы матрицы идеальными значениями.
    def draw_finder_pattern(r_start, c_start):
        # Внешняя черная рамка 7x7
        matrix[r_start:r_start+7, c_start:c_start+7] = 0
        # Внутренняя белая рамка 5x5
        matrix[r_start+1:r_start+6, c_start+1:c_start+6] = 255
        # Центральный черный квадрат 3x3
        matrix[r_start+2:r_start+5, c_start+2:c_start+5] = 0

    # Левый верхний угол
    draw_finder_pattern(0, 0)
    # Правый верхний угол
    draw_finder_pattern(0, grid_size - 7)
    # Левый нижний угол
    draw_finder_pattern(grid_size - 7, 0)

    # (Опционально) Alignment pattern - маленький квадратик 5х5 ближе к правому нижнему углу
    # У версии 2 (25х25) его центр находится на координатах (18, 18)
    if grid_size == 25:
        r, c = 16, 16 # Координаты левого верхнего угла этого паттерна 5х5
        matrix[r:r+5, c:c+5] = 0      # Внешняя черная
        matrix[r+1:r+4, c+1:c+4] = 255 # Внутренняя белая
        matrix[r+2, c+2] = 0          # Центральная черная точка

    # 4. ОТРИСОВКА В ВЫСОКОМ РАЗРЕШЕНИИ
    # Используем INTER_NEAREST, чтобы квадратики остались резкими, без размытия
    out_img = cv2.resize(matrix, (output_size, output_size), interpolation=cv2.INTER_NEAREST)

    # Добавляем "тихую зону"
    padded = cv2.copyMakeBorder(out_img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)

    return padded

# --- Пример использования ---
if __name__ == "__main__":
    img = cv2.imread("/work/lenta_cv/vision/_img/qr_3.jpg") # Ваш исходный кроп

    if img is not None:
        # Для начала пробуем версию 2 (25x25)
        perfect_qr = reconstruct_qr_hardcoded(img, grid_size=25)

        cv2.imwrite("perfect_hardcoded_qr.jpg", perfect_qr)
        print("Код восстановлен и сохранен!")

        decoded = decode(perfect_qr)
        if decoded:
            print(f"🎉 УРА! Прочитано: {decoded[0].data.decode('utf-8')}")
        else:
            print("❌ Пока не читается. Проверьте сохраненную картинку.")