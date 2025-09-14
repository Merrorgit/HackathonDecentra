# 🏦 OCR для банковских контрактов. Decentrathon 4.0. Второй кейс от BCC.

**Интеллектуальная OCR-система нового поколения для обработки многостраничных банковских контрактов в формате PDF**

## 🎯 Описание проекта

Это решение для хакатона - система OCR 2.0, которая мы надеемся превосходит базовый Tesseract в обработке банковских контрактов. Система извлекает текст со ВСЕХ страниц PDF документов, структурирует данные в JSON формат и использует LLM для пост-обработки.

### 🚀 Ключевые возможности

- **Только PDF документы** - специализированная обработка PDF контрактов
- **Многостраничная обработка** - извлечение текста со всех страниц документа (до 20 страниц)
- **Продвинутый OCR** - PaddleOCR с русской моделью для лучшего распознавания контрактов
- **Извлечение структурированных данных** - автоматическое определение ключевых полей
- **LLM пост-обработка** - использование gemma3:1b для нормализации и структуризации данных
- **Веб-интерфейс** - удобный Streamlit интерфейс

### 📊 Извлекаемые поля

- **contract_date** - дата заключения (дата заключения контракта)
- **expiration_date** - дата окончания (срок действия контракта, договора)  
- **counterparty** - контрагент (наименование иностранного контрагента)
- **country** - страна (страна контрагента (инопартнера))
- **contract_amount** - сумма контракта (число без валюты)
- **contract_currency** - валюта контракта
- **payment_currency** - валюта платежа

## 🛠️ Технический стек

- **OCR**: PaddleOCR 
- **LLM**: Ollama + gemma3:1b (ограничение по техническим возможностям, возможна загрузка более умной и точной модели LLM)
- **PDF обработка**: pdf2image + PyMuPDF (fallback)
- **UI**: Streamlit
- **Computer Vision**: OpenCV, PIL

## 📦 Установка

### 1. Установка зависимостей Python

```bash
```markdown
# 🏦 OCR 2.0 для банковских контрактов (PDF only)

## 👥 Команда
- Команда: «sudo rm -rf /»
- Участники: Дмитрий Палиев, Михаил Жеребцов

## 🛠 Краткий стек технологий
- Streamlit (веб-интерфейс)
- PyMuPDF (прямое извлечение текста из PDF с порядком блоков)
- PaddleOCR (русский язык; OCR fallback)
- OpenCV + PIL (предобработка изображений, CLAHE/deskew)
- Ollama + gemma3:1b (LLM для извлечения 8 полей в JSON)

Извлекаемые поля JSON:
- contract_number, contract_date, expiration_date, counterparty, country,
  contract_amount, contract_currency, payment_currency

## 🚀 Как запустить
1) Подготовка окружения (fish shell):
```fish
cd /home/mikhail/projects/hackathon
python3 -m venv venv
source venv/bin/activate.fish
pip install -r requirements.txt
```
2) Убедитесь, что запущен Ollama и загружена модель (которую вы выбрали):
```bash
ollama serve
ollama pull gemma3:1b 
```
3) Запуск приложения:
```fish
source venv/bin/activate.fish
python3 -m streamlit run src/app.py
```
Приложение будет доступно на http://localhost:8501

(Опционально, для сервера)
```fish
python3 -m streamlit run src/app.py --server.headless true --server.port 8501
```

## 📂 Структура (кратко)
```
src/
  app.py              # Streamlit UI (PDF only, настройки OCR в сайдбаре)
  ollama_client.py    # Взаимодействие с Ollama 
  pipeline/
    pdf_utils.py      # Извлечение текста: direct → OCR fallback
    ocr.py            # PaddleOCR + группировка строк
    preprocess.py     # Предобработка (deskew, CLAHE, резкость)
requirements.txt
README.md
.gitignore
```

## ℹ️ Примечания
- Прямой текст из PDF используется, если он не «битый». Иначе включается OCR с усиленной предобработкой (по настройке).
- Для лучшего качества OCR можно повысить DPI (до 400) и включить «Усиленный режим для сканов» в сайдбаре.
