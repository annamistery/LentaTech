import cv2
import numpy as np

def read_codes_opencv(image: np.ndarray) -> list[dict]:
    """
    Распознавание штрихкодов и QR-кодов встроенными средствами OpenCV.
    Не требует pyzbar и системных библиотек.
    """
    results = []

    # --- 1. Поиск линейных штрихкодов (EAN-13, UPC и т.д.) ---
    barcode_detector = cv2.barcode.BarcodeDetector()
    # retval - успешность, decoded_info - сами цифры, decoded_type - формат
    retval, decoded_info, decoded_type = barcode_detector.detectAndDecode(image)

    if retval:
        # OpenCV возвращает кортежи/списки, перебираем их
        for i in range(len(decoded_info)):
            if decoded_info[i]:  # Отсекаем пустые считывания
                # В зависимости от сборки OpenCV, decoded_type может возвращать числа
                # или строки. Для простоты пишем просто "BARCODE"
                results.append({
                    "data": decoded_info[i],
                    "type": "BARCODE"
                })

    # --- 2. Поиск QR-кодов ---
    qr_detector = cv2.QRCodeDetector()
    retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(image)

    if retval:
        for info in decoded_info:
            if info:
                results.append({
                    "data": info,
                    "type": "QRCODE"
                })

    return results