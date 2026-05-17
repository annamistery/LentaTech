# запуск сервера  python3 -m llama_cpp.server   --model ZwZ-4B.Q8_0.gguf   --clip_model_path mmproj-ZwZ-4B-Q8_0.gguf --chat_format qwen2.5-vl  --n_ctx 4096   --host 0.0.0.0   --port 8000

from openai import OpenAI
import base64

# Кодируем вашу картинку (ценник)
filename = "../_img/2.jpg"
with open(filename, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

prompt_text = """
Ты система распознавания ценников.
Верни название и стоимость товара с ценника. 
Цена в формате рубли-копейки. Копейки обозначены более мелким шрифтом после рублей чуть выше - так обычно математики обозначают возведение в степень.

Ответ строго в JSON формате, например: {"product_name": "сыр гауда", "price": "399.00"}
"""

# Подключаемся к локальному серверу llama.cpp
client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-no-key-required")

response = client.chat.completions.create(
    model="gpt-3.5-turbo", # Имя не важно для локального сервера
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
