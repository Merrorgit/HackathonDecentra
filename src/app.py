import streamlit as st
import sys
import os
from pathlib import Path

# Добавляем корневую директорию в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image
from src.pipeline.ocr import run_ocr
from src.pipeline.pdf_utils import extract_all_pages_text, is_pdf_file
from src.ollama_client import extract_fields

st.set_page_config(page_title="OCR Decentrathon")
st.title("Team Sudo rm -rf /. Case 2. OCR.")
st.markdown("*Загрузите документ в формате PDF для извлечения данных*")

# Поддержка только PDF с ограничениями
uploaded = st.file_uploader(
    "Загрузить PDF контракт", 
    type=["pdf"]
)

if uploaded:
    # Проверяем размер файла
    if uploaded.size > 10 * 1024 * 1024:  # 10 МБ
        st.error("❌ Файл слишком большой! Максимальный размер: 10 МБ")
        st.stop()
    
    file_bytes = uploaded.read()
    
    # Проверяем, что это PDF файл
    if not is_pdf_file(file_bytes):
        st.error("❌ Загруженный файл не является PDF документом!")
        st.stop()
        st.stop()
    
    st.info("📄 PDF документ загружен. Готов к обработке...")
    st.success(f"📊 Размер файла: {len(file_bytes)} байт")

    # Параметры OCR из сайдбара
    with st.sidebar:
        st.header("⚙️ Настройки OCR")
        force_ocr = st.checkbox("Игнорировать встроенный текст (Force OCR)", value=False)
        strong_mode = st.checkbox("Усиленный режим для сканов", value=False)
        dpi = st.slider("DPI для OCR", min_value=200, max_value=400, step=50, value=300)
        max_pages = st.number_input("Макс. страниц", min_value=1, max_value=50, value=10, step=1)

    if st.button("🔍 Запустить OCR на всех страницах", type="primary"):
        with st.spinner("Обрабатываем все страницы PDF документа..."):
            try:
                # Запускаем извлечение текста со всеми настройками
                text = extract_all_pages_text(
                    file_bytes,
                    dpi=int(dpi),
                    max_pages=int(max_pages),
                    force_ocr=bool(force_ocr),
                    strong_mode=bool(strong_mode),
                )
                
                if not text.strip():
                    st.error("❌ Не удалось извлечь текст из документа!")
                    st.stop()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📝 Извлеченный текст")
                    st.text_area("OCR Result", value=text, height=400, key="ocr_text")
                
                with col2:
                    st.subheader("🏷️ Структурированные данные")
                    
                    with st.spinner("Анализируем данные с помощью LLM..."):
                        # Поля для извлечения из банковских контрактов
                        fields = [
                            "contract_number",        # № контракта
                            "contract_date",          # дата заключения (дата заключения контракта)
                            "expiration_date",        # дата окончания (срок действия контракта, договора)
                            "counterparty",           # контрагент (наименование иностранного контрагента)
                            "country",                # страна (страна контрагента (инопартнера))
                            "contract_amount",        # сумма контракта
                            "contract_currency",      # валюта контракта
                            "payment_currency"        # валюта платежа
                        ]
                        
                        parsed = extract_fields(text, fields)
                        st.json(parsed)
                    
                    # Показываем метрики качества
                    if text.strip():
                        # Подсчитываем количество страниц
                        pages_count = text.count("=== СТРАНИЦА")
                        st.success(f"✅ Обработано страниц: {pages_count}")
                        st.info(f"📊 Всего символов: {len(text)}")
                        extracted_fields = sum(1 for v in parsed.values() if v is not None)
                        st.info(f"🎯 Извлечено полей: {extracted_fields}/{len(fields)}")
                    
            except Exception as e:
                st.error(f"❌ Ошибка при обработке: {str(e)}")
                st.error("Проверьте, что модель gemma3:1b запущена в Ollama")

# Оставляем только настройки OCR в сайдбаре (см. выше внутри блока uploaded)
