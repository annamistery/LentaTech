import cv2

from vision.utils.base64_converter import ImageConverter
from vision.utils.prompt_build import build_prompt_from_yaml
from vision.cpp_llm.llama_cpp_client import LlamaCPPClient
from vision.qr_bar_code.pyzbar_simple import read_barcode_basic

def cpp_ocr(img_bgr):
    try:
        # увеличим разрешение
        #processed_img  = simple_ai_upscale(img_bgr, "../upscaler/bin/EDSR_x3.pb")
        processed_img = cv2.resize(img_bgr, None, fx=3, fy=3)

        # 2. Конвертация в base64
        base64_img = ImageConverter.cv2_to_base64(processed_img)

        # 3. Подготовка промпта (из YAML)
        prompt = build_prompt_from_yaml("../cpp_llm/prompt_schema.yaml")

        # 4. Отправка на сервер
        print("Отправка запроса на локальный сервер...")
        llm_client = LlamaCPPClient(base_url="http://localhost:8000/v1")
        result = llm_client.extract_json_from_image(prompt, base64_img)

        # 5. читаем штрихкод
        barcodes = read_barcode_basic(processed_img)

        # Результат
        print("\n--- Результат ---")
        for item in result:
            print(f"{item}: {result[item]}")

        if barcodes:
            print(f"Найдено: {barcodes}")
        else:
            print("Штрихкод не найден.")

    except Exception as e:
        print(f"[Ошибка] {e}")

if __name__ == '__main__':
    img_path = "../_img/qr.jpg"
    img = cv2.imread(img_path)
    cpp_ocr(img)
