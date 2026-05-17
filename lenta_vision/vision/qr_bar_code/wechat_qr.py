

import os
import cv2


def read_qr_wechat_local(image_path: str, models_dir: str = "wechat_models"):
    """
    Распознает QR-код с помощью локальных моделей WeChat.
    """
    print(f"\n--- Запуск WeChat QR (Local) для {image_path} ---")

    # 1. Формируем пути к локальным файлам
    det_prototxt = os.path.join(models_dir, "detect.prototxt")
    det_model = os.path.join(models_dir, "detect.caffemodel")
    sr_prototxt = os.path.join(models_dir, "sr.prototxt")
    sr_model = os.path.join(models_dir, "sr.caffemodel")

    # Быстрая проверка, что вы положили файлы куда надо
    if not os.path.exists(det_model):
        print(f"❌ Ошибка: Файлы моделей не найдены в папке '{models_dir}'.")
        return

    # 2. Инициализируем нейросеть
    detector = cv2.wechat_qrcode_WeChatQRCode(det_prototxt, det_model, sr_prototxt, sr_model)

    # 3. Читаем изображение
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Ошибка: Файл картинки не найден.")
        return

    # Добавляем небольшую белую рамку для стабильности
    padded = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # 4. Детекция и декодирование
    res, points = detector.detectAndDecode(padded)

    if res:
        for text in res:
            print(f"🎉 УСПЕХ (WeChat CNN): {text}")
    else:
        print("❌ ПРОВАЛ: Код не прочитан.")

if __name__ == "__main__":
    # Тестируем на вашем кропе
    read_qr_wechat_local("/work/lenta_cv/vision/_img/qr.jpg", models_dir="models")