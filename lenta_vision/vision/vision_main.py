from pathlib import Path

import cv2
import logging
import numpy as np
from typing import Dict, Any, Tuple

from vision.cpp_llm.llama_cpp_client import LlamaCPPClient
from vision.modules.blur_check import get_blur_score_robust, is_not_blurred
from vision.modules.hsv_color_check import detect_price_tag_color
from vision.qr_bar_code.cv_codes import read_codes_opencv
from vision.utils.base64_converter import ImageConverter
from vision.utils.prompt_build import build_prompt_from_yaml

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("VisionPipeline")


# ==========================================
# Вспомогательные функции (Валидаторы)
# ==========================================

def validate_ocr_data(ocr_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Проверяет наличие обязательных полей от Qwen:
    - product_name (не пустой)
    - хотя бы одна цена (price_default, price_card, price_discount)
    """
    if not isinstance(ocr_data, dict):
        return False, "LLM вернула невалидный JSON (не словарь)"

    # Проверка имени
    has_name = bool(ocr_data.get("product_name"))

    # Проверка цен (ищем хотя бы одну не None и не пустую)
    has_price = any([
        bool(ocr_data.get("price_default")),
        bool(ocr_data.get("price_card")),
        bool(ocr_data.get("price_discount"))
    ])

    if not has_name:
        return False, "Отсутствует product_name"
    if not has_price:
        return False, "Отсутствует любая из цен (default, card, discount)"

    return True, "ok"


# ==========================================
# ГЛАВНАЯ ФУНКЦИЯ ПАЙПЛАЙНА
# ==========================================

def vision_main(img_bgr: np.ndarray, config_dict: dict  = {}) -> dict:
    """
    Главный конвейер обработки изображения ценника.
    Принимает изображение и словарь с настройками.
    """
    # 0. Читаем конфиги (с дефолтными значениями на случай их отсутствия)
    BLUR_THRESHOLD = config_dict.get("min_blur_score", 8.0)
    COLOR_THRESHOLD = config_dict.get("color_threshold", 0.25)
    LLM_BASE_URL = config_dict.get("llm_base_url", "http://localhost:8000/v1")
    CPP_FOLDER = config_dict.get("cpp_folder", "cpp_llm")
    PROMPT_CONFIG = config_dict.get("prompt_congig_file", "prompt_schema.yaml")
    UPSCALE_FX = config_dict.get("upscale_fx", 3)

    PROMPT_PATH = Path(CPP_FOLDER, PROMPT_CONFIG)

    payload = {}

    try:
        # 1. ПРОВЕРКА РАЗМЫТИЯ
        is_sharp, blur_score = is_not_blurred(img_bgr, BLUR_THRESHOLD)
        #payload["blur_score"] = round(blur_score, 2)

        if not is_sharp:
            logger.warning(f"Брак кадра (Размытие): Score {blur_score:.2f} < {BLUR_THRESHOLD}")
            return {"processed": False, "reason": "blur", "payload": payload}

        # 2. ПРОВЕРКА ЦВЕТА
        tag_color = detect_price_tag_color(img_bgr, COLOR_THRESHOLD)
        payload["color"] = tag_color

        # Если нужно отсеивать "other" - раскомментируйте этот блок
        # if tag_color == "other" and config_dict.get("drop_other_colors", True):
        #     logger.warning("Брак кадра (Цвет): Не похоже на ценник (other)")
        #     return {"processed": False, "reason": "color_other", "payload": payload}

        # 3. ПОДГОТОВКА И ОТПРАВКА В QWEN Vision
        logger.info(f"Кадр прошел фильтры (Score: {blur_score:.1f}, Цвет: {tag_color}). Отправка в LLM...")

        # Апскейл перед OCR
        processed_img = cv2.resize(img_bgr, None, fx=UPSCALE_FX, fy=UPSCALE_FX, interpolation=cv2.INTER_CUBIC)
        base64_img = ImageConverter.cv2_to_base64(processed_img)
        prompt = build_prompt_from_yaml(PROMPT_PATH)

        llm_client = LlamaCPPClient(base_url=LLM_BASE_URL)
        llm_data = llm_client.extract_json_from_image(prompt, base64_img)

        # 4. ВАЛИДАЦИЯ ОБЯЗАТЕЛЬНЫХ ПОЛЕЙ
        is_valid, validation_msg = validate_ocr_data(llm_data)

        if not is_valid:
            logger.error(f"Брак OCR: {validation_msg}")
            payload.update(llm_data if isinstance(llm_data, dict) else {})
            return {"processed": False, "reason": "ocr_validation_failed", "payload": payload}

        logger.info("OCR валидация пройдена успешно.")
        payload.update(llm_data) # Добавляем данные от Qwen в общий payload

        # 5. ПОИСК ШТРИХКОДОВ/QR (Только для хороших кадров)
        logger.info("Запуск поиска штрихкодов и QR...")
        # Передаем апскейл-версию, так как cv2-детекторам нужно больше пикселей
        codes_found = read_codes_opencv(processed_img)

        if codes_found:
            logger.info(f"Найдено кодов: {len(codes_found)}")
            # Если нужно, можно добавить логику выбора, если найдено несколько.
            # Пока запишем список найденных сырых кодов в payload
            payload["detected_codes"] = codes_found
        else:
            payload["detected_codes"] = []

        # 6. УСПЕШНЫЙ ФИНАЛ
        logger.info("Пайплайн успешно завершен!")
        return {
            "processed": True,
            "payload": payload
        }

    except Exception as e:
        logger.exception("Критическая ошибка в пайплайне vision_main:")
        return {
            "processed": False,
            "reason": "process_error",
            "details": str(e),
            "payload": payload # Возвращаем то, что успели собрать до падения
        }

if __name__ == '__main__':
    img_path = "_img/bad_2.jpg"
    img = cv2.imread(img_path)
    answer = vision_main(img)
    print(answer)