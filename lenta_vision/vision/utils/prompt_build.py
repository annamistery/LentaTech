
import yaml

# Функция загрузки и конвертации YAML в текстовый промпт
def build_prompt_from_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Собираем системный промпт
    prompt = config['system_instruction'] + "\n\nСТРУКТУРА JSON:\n{\n"

    # Динамически добавляем поля и их описания
    for field, attributes in config['fields'].items():
        prompt += f'  "{field}": <{attributes["type"]}> - {attributes["description"]}\n'

    prompt += "}"
    return prompt