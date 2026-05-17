import cv2

from vision._examples.cpp_client_server_ex import cpp_ocr
from vision.modules.blur_check import get_blur_score, get_blur_score_robust


def run_synthetic_blur_test(img):
    print(f"=== Запуск синтетического теста размытия ===")

    # Набор ядер для размытия (размер должен быть нечетным числом)
    # 1 - чистый кадр, 7 - легкий смаз, 15 - заметное мыло, 25 и 35 - сильное искажение
    blur_kernels = [1, 7, 15, 25, 35]

    for k in blur_kernels:
        # 1. Применяем синтетическое размытие
        blurred_img = img if k == 1 else cv2.GaussianBlur(img, (k, k), 0)

        # 2. Получаем математическую оценку резкости
        laplacian_score = get_blur_score_robust(blurred_img)

        print(f"► Размытие: ядро {k}x{k} | Laplacian Score: {laplacian_score:.2f}")

        cpp_ocr(blurred_img)

if __name__ == '__main__':
    img_path = "../_img/1.jpg"
    image = cv2.imread(img_path)
    run_synthetic_blur_test(image)