# src/ollama_client.py
import os, json, re, requests
from typing import List, Dict, Any, Union

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

def generate(prompt: str, model: str = None) -> str:
    """Генерирует ответ от LLM через Ollama"""
    model = model or DEFAULT_MODEL
    
    try:
        if HAS_OLLAMA:
            response = ollama.generate(model=model, prompt=prompt)
            # Извлекаем текст из ответа - у объекта есть атрибут response
            if hasattr(response, 'response'):
                return response.response
            elif isinstance(response, dict):
                return response.get("response", "")
            else:
                return str(response)
        else:
            # Fallback на HTTP API
            r = requests.post(
                f"{OLLAMA_HOST}/api/generate", 
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=60
            )
            r.raise_for_status()
            return r.json().get("response", "")
    except Exception as e:
        raise Exception(f"Ошибка генерации от модели {model}: {str(e)}")

def extract_fields(ocr_text: str, fields: List[str]) -> Dict[str, Any]:
    """
    Извлекает структурированные поля из OCR текста используя LLM
    
    Args:
        ocr_text: Текст, полученный от OCR
        fields: Список полей для извлечения
        
    Returns:
        Словарь с извлеченными полями
    """
    if not ocr_text.strip():
        return {k: None for k in fields}
    
    prompt = f"""Ты строгий экстрактор JSON данных из банковских контрактов и договоров. 
Извлеки следующие поля из OCR текста: {fields}

Поля и их значения:
- contract_number: № контракта (номер контракта или договора)
- contract_date: дата заключения (дата заключения контракта) в формате YYYY-MM-DD
- expiration_date: дата окончания (срок действия контракта, договора) в формате YYYY-MM-DD  
- counterparty: контрагент (наименование иностранного контрагента)
- country: страна (страна контрагента/инопартнера)
- contract_amount: сумма контракта как число (без валюты)
- contract_currency: валюта контракта (USD, EUR, RUB и т.д.)
- payment_currency: валюта платежа (USD, EUR, RUB и т.д.)

Правила:
- Даты нормализуй в формат YYYY-MM-DD
- Суммы конвертируй в float числа без валюты
- Валюты указывай как коды (USD, EUR, RUB, CNY и т.д.)
- Номер контракта ищи по ключевым словам: "№", "номер", "contract", "договор"
- Если поле отсутствует или неопределено — ставь null
- Выводи ТОЛЬКО валидный JSON без дополнительного текста
- Названия организаций приводи к читаемому виду

OCR_TEXT:
{ocr_text[:2000]}"""  # Ограничиваем длину для экономии токенов

    try:
        response = generate(prompt)
        return _parse_json_response(response, fields)
    except Exception as e:
        print(f"Ошибка извлечения полей: {e}")
        return {k: None for k in fields}

def _parse_json_response(response: str, fields: List[str]) -> Dict[str, Any]:
    """Парсит JSON ответ от LLM с fallback стратегиями"""
    
    # 1. Убираем markdown блоки
    cleaned_response = response.strip()
    if cleaned_response.startswith('```json'):
        cleaned_response = cleaned_response[7:]  # Убираем ```json
    if cleaned_response.startswith('```'):
        cleaned_response = cleaned_response[3:]   # Убираем ```
    if cleaned_response.endswith('```'):
        cleaned_response = cleaned_response[:-3]  # Убираем ``` в конце
    
    cleaned_response = cleaned_response.strip()
    
    # 2. Прямой парсинг
    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        pass
    
    # 3. Поиск JSON блока в тексте
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_response, re.DOTALL)
    if json_match:
        candidate = json_match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        
        # 4. Исправление одинарных кавычек
        try:
            fixed = candidate.replace("'", '"')
            # Исправляем trailing commas
            fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
    
    # 5. Если ничего не получилось - возвращаем пустой результат
    return {k: None for k in fields}
