# пример запуска сервера  python3 -m llama_cpp.server   --model ZwZ-4B.Q8_0.gguf   --clip_model_path mmproj-ZwZ-4B-Q8_0.gguf --chat_format qwen2.5-vl  --n_ctx 4096  --host 0.0.0.0   --port 8000
# из конфига python -m llama_cpp.server --config_file server_config.json

import logging

logger = logging.getLogger(__name__)


from openai import OpenAI
import json

class LlamaCPPClient:
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "sk-no-key"):
        """
        Инициализация клиента для локального сервера llama-cpp-python.
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.default_model = "local-model" # Имя не имеет значения для llama.cpp

    def extract_json_from_image(self, prompt: str, base64_image: str, temperature: float = 0.1) -> dict:
        """
        Отправляет промпт и изображение на сервер и пытается распарсить JSON.

        :param prompt: Текстовая инструкция
        :param base64_image: Изображение в формате base64
        :param temperature: Температура генерации (0.1 для предсказуемости OCR)
        :return: Словарь (dict) с результатами или None в случае ошибки
        """
        try:
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                temperature=temperature
            )

            raw_result = response.choices[0].message.content
            return self._parse_json(raw_result)

        except Exception as e:
            logger.error(f"[LLM Client Error] Ошибка при обращении к серверу: {e}")
            return {"error": str(e)}

    def _parse_json(self, text: str) -> dict:
        """
        Внутренний метод для очистки ответа модели от возможного мусора (маркдауна)
        и безопасного парсинга JSON.
        """
        text = text.strip()
        # Если модель обернула ответ в маркдаун блоки ```json ... ```
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            logger.error(f"[JSON Error] Не удалось распарсить ответ модели. Сырой текст:\n{text}")
            return {"raw_text": text}