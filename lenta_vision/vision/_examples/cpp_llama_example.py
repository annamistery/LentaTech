import os
import json
import base64

import cv2

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

# --- Настройки ---
# Укажите путь к вашей GGUF модели (должна поддерживать Vision, например Qwen2-VL)
model_path = "../cpp_llm/bin/ZwZ-4B.Q8_0.gguf"
mmproj_path = "../cpp_llm/bin/mmproj-ZwZ-4B-Q8_0.gguf"

# Функция для кодирования ценника в формат Base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"

# Инициализируем обработчик изображений
chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path)
# Путь к кропу ценника для теста
IMAGE_PATH = "../_img/2.jpg"



def main():
    print("⏳ Загрузка модели... (это может занять время при первом запуске)")

    # Инициализация модели
    llm = Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=4096,             # Контекстное окно
        n_gpu_layers=0,         # 0 = CPU, -1 = GPU (все слои)
        n_threads=4,
        verbose=True,
    )

    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Файл изображения не найден: {IMAGE_PATH}")
        return

    # Загрузка изображения
    image_64 = image_to_base64(IMAGE_PATH)

    # --- Промпт по твоему запросу ---
    prompt_text = """
    Ты система распознавания ценников.
    Верни название и стоимость товара с ценника. 
    Цена в формате рубли-копейки. Копейки обозначены более мелким шрифтом после рублей чуть выше - так обычно математики обозначают возведение в степень.
    
    Ответ строго в JSON формате: {"product_name": "...", "price": "..."}
    """

    # Формирование сообщения для мультимодальной модели
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_64}"
                    }
                }
            ]
        }
    ]


    print("🚀 Запуск инференса...")

    try:
        # Запуск генерации
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.1,   # Низкая температура для точности
            top_p=0.9,
            repeat_penalty=1.1
        )

        # Вывод структурированного результата
        print(output["choices"]["message"]["content"])


    except Exception as e:
            print(f"\n❌ Ошибка при инференсе: {e}")

if __name__ == "__main__":
    main()