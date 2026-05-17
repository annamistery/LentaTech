import cv2
import numpy as np
from pyzbar.pyzbar import decode

def order_points(pts):
    """Сортирует 3 найденные точки маркеров: Верх-Лево, Верх-Право, Низ-Лево"""
    # Сортируем по X
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # Левые две точки
    leftMost = xSorted[:2, :]
    # Правая точка (обычно это Верх-Право или Низ-Право, но так как их 3, логика чуть хитрее)
    # Надежнее найти прямой угол. Точка, расстояние от которой до двух других максимально похоже, это Верх-Лево.

    # Для простоты: найдем точку с минимальной суммой координат (Верх-Лево)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]

    # Исключим tl из списка
    rem = np.delete(pts, np.argmin(s), axis=0)

    # Из оставшихся двух та, что выше (меньше Y) - это Верх-Право, другая - Низ-Лево
    if rem[0][1] < rem[1][1]:
        tr, bl = rem[0], rem[1]
    else:
        tr, bl = rem[1], rem[0]

    return np.array([tl, tr, bl], dtype="float32")

def reconstruct_warped_qr(image: np.ndarray, grid_size: int = 25, out_size: int = 500):
    """
    1. Ищет 3 маркера (матрешки).
    2. Выравнивает перспективу (Аффинное преобразование).
    3. Делает сеточное семплирование.
    """
    # 0. ПРЕДОБРАБОТКА И АДАПТИВНАЯ БИНАРИЗАЦИЯ
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Адаптивная бинаризация: смотрит на окно 21x21 пиксель
    # Спасает те самые внутренние белые рамки маркеров, которые убивает Оцу
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, # Размер окна (можно поиграть: 11, 21, 31)
        2   # Константа вычитания
    )

    # 1. Убираем мелкие черные точки и разрываем тонкие черные мостики
    # MORPH_CLOSE (при белом фоне 255) сначала "наращивает" белое, съедая тонкие черные линии,
    # а потом сужает обратно.
    kernel_close = np.ones((3, 3), np.uint8)
    clean_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # 2. Убираем мелкие белые точки внутри черных областей
    kernel_open = np.ones((3, 3), np.uint8)
    clean_binary = cv2.morphologyEx(clean_binary, cv2.MORPH_OPEN, kernel_open)

    # Сохраните и посмотрите на clean_binary!
    cv2.imwrite("debug_morphology.jpg", clean_binary)

    # Теперь ищем контуры по ОЧИЩЕННОЙ картинке
    contours, hierarchy = cv2.findContours(clean_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    marker_centers = []

    if hierarchy is not None:
        for i, h in enumerate(hierarchy[0]):
            # h = [Next, Previous, First_Child, Parent]
            child = h[2]
            if child != -1:
                grandchild = hierarchy[0][child][2]
                if grandchild != -1:
                    # Нашли 3 уровня вложенности! Это кандидат в маркер.
                    # Вычисляем центр контура
                    M = cv2.moments(contours[i])
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        marker_centers.append([cX, cY])

    # Если нашли больше или меньше 3 маркеров - алгоритм запутался (обычно из-за теней)
    if len(marker_centers) < 3:
        print(f"❌ Найдено маркеров: {len(marker_centers)}. Нужно ровно 3. Выравнивание невозможно.")
        return None

    # Берем первые 3 (в идеале нужно фильтровать по площади, если их больше)
    pts = np.array(marker_centers[:3], dtype="float32")
    pts_ordered = order_points(pts)

    # 2. ВЫРАВНИВАНИЕ ПЕРСПЕКТИВЫ (Аффинное преобразование по 3 точкам)
    # Маркеры находятся не на самом краю, а отступают на 3.5 модуля от края.
    # Высчитываем координаты идеальных маркеров на нашем новом холсте out_size x out_size
    cell_size = out_size / grid_size
    offset = 3.5 * cell_size

    ideal_tl = [offset, offset]
    ideal_tr = [out_size - offset, offset]
    ideal_bl = [offset, out_size - offset]

    pts_ideal = np.float32([ideal_tl, ideal_tr, ideal_bl])

    # Вычисляем матрицу поворота и растяжения
    matrix = cv2.getAffineTransform(pts_ordered, pts_ideal)
    # Натягиваем кривую бинарную картинку на ровный квадрат
    warped = cv2.warpAffine(binary, matrix, (out_size, out_size), flags=cv2.INTER_NEAREST, borderValue=255)

    cv2.imwrite("debug_warped.jpg", warped) # Сохраним посмотреть, как оно выровнялось

    # 3. СЕТОЧНОЕ СЕМПЛИРОВАНИЕ И РИСОВАНИЕ ХАРДКОД-МАРКЕРОВ
    # (Здесь используется логика из нашего предыдущего скрипта)
    # ... Для экономии места я не дублирую тут весь цикл отрисовки из прошлой версии,
    # он будет идти по уже идеально ровной переменной `warped`.

    # Для быстрого теста давайте просто попробуем прочитать саму выровненную картинку
    padded = cv2.copyMakeBorder(warped, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    return padded

if __name__ == "__main__":
    img = cv2.imread("/work/lenta_cv/vision/_img/qr_3.jpg")
    if img is not None:
        result = reconstruct_warped_qr(img)
        if result is not None:
            decoded = decode(result)
            print(f"Прочитано: {decoded}")