# src/pipeline/pdf_utils.py
import io
from typing import List
from PIL import Image
import pdf2image
import fitz  # PyMuPDF

def pdf_to_images(pdf_bytes: bytes, dpi: int = 200, max_pages: int = 10) -> List[Image.Image]:
    """
    Конвертирует PDF в список PIL изображений.
    
    Args:
        pdf_bytes: PDF файл в виде байтов
        dpi: Разрешение для конвертации (200 для экономии памяти)
        max_pages: Максимальное количество страниц для обработки
        
    Returns:
        Список PIL.Image объектов
    """
    images = []
    
    try:
        # Используем pdf2image (через poppler) с ограниченным DPI
        images = pdf2image.convert_from_bytes(
            pdf_bytes,
            dpi=dpi,
            first_page=1,
            last_page=min(max_pages, 20),  # уменьшаем лимит страниц
            fmt='RGB',
            thread_count=1  # ограничиваем потоки
        )
        return images[:max_pages]
    
    except Exception as e:
        # Fallback на PyMuPDF
        try:
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            images = []
            
            for page_num in range(min(len(pdf_doc), max_pages)):
                page = pdf_doc[page_num]
                
                # Конвертируем в изображение с умеренным разрешением
                mat = fitz.Matrix(dpi/72, dpi/72)  # 72 DPI - базовое разрешение PDF
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Создаем PIL Image
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
            
            pdf_doc.close()
            return images
            
        except Exception as fallback_error:
            raise Exception(f"Не удалось конвертировать PDF: {e}, fallback error: {fallback_error}")

def extract_all_pages_text(pdf_bytes: bytes, dpi: int = 300, max_pages: int = 20, *, force_ocr: bool = False, strong_mode: bool = False) -> str:
    """
    Извлекает текст со всех страниц PDF с приоритетом прямого извлечения (PyMuPDF)
    и fallback на OCR только при необходимости (пустые/сканированные страницы).

    Args:
        pdf_bytes: PDF файл в виде байтов
        dpi: DPI для растеризации при OCR (используется только если нужен OCR)
        max_pages: Максимальное количество страниц для обработки

    Returns:
        Объединенный текст со всех страниц, с разделителями по страницам
    """
    from src.pipeline.ocr import run_ocr
    import gc

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        # Если PDF поврежден для fitz — пробуем через старый путь
        return _extract_all_pages_text_via_ocr_only(pdf_bytes, dpi=dpi, max_pages=max_pages)

    all_text: List[str] = []
    total_pages = min(len(doc), max_pages)
    for i in range(total_pages):
        try:
            page = doc[i]

            # 1) Пробуем напрямую вытащить текст (лучшее для цифровых PDF), если не форсируем OCR
            if not force_ocr:
                direct_text = _direct_text_from_page(page)
                if direct_text and len(direct_text.strip()) >= 25:
                    # Оцениваем качество прямого текста
                    if not _looks_corrupted_text(direct_text):
                        print(f"📄 Стр. {i+1}/{total_pages}: direct PDF text")
                        all_text.append(f"=== СТРАНИЦА {i+1} ===\n{direct_text.strip()}")
                        continue
                    else:
                        print(f"🖼️ Стр. {i+1}/{total_pages}: direct text looks corrupted -> OCR fallback")

            # 2) Fallback: растеризуем страницу и используем OCR
            use_dpi = max(300, dpi)
            print(f"🖼️ Стр. {i+1}/{total_pages}: OCR fallback (dpi={use_dpi})")
            mat = fitz.Matrix(use_dpi / 72.0, use_dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_data = pix.tobytes("png")
            pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
            if strong_mode:
                # Сразу используем усиленную предобработку
                try:
                    from src.pipeline.preprocess import preprocess_pil_image
                    pil_strong = preprocess_pil_image(pil_img, strong=True)
                    page_text = run_ocr(pil_strong, do_preprocess=False)
                except Exception:
                    page_text = run_ocr(pil_img)
            else:
                page_text = run_ocr(pil_img)
                if not page_text.strip():
                    # Вторая попытка: более сильная предобработка
                    try:
                        from src.pipeline.preprocess import preprocess_pil_image
                        pil_strong = preprocess_pil_image(pil_img, strong=True)
                        page_text = run_ocr(pil_strong, do_preprocess=False)
                    except Exception:
                        pass
            pil_img.close()
            del pix, img_data
            gc.collect()

            if page_text.strip():
                all_text.append(f"=== СТРАНИЦА {i+1} ===\n{page_text}")
            else:
                all_text.append(f"=== СТРАНИЦА {i+1} ===\n")

        except Exception as e:
            print(f"⚠️ Ошибка на странице {i+1}: {e}")
            all_text.append(f"=== СТРАНИЦА {i+1} ===\n")
        finally:
            gc.collect()

    doc.close()
    return "\n\n".join(all_text)


def _direct_text_from_page(page: "fitz.Page") -> str:
    """Возвращает текст страницы в читабельном порядке по блокам.
    Используем blocks для лучшего сохранения строк и порядка чтения.
    """
    try:
        blocks = page.get_text("blocks") or []
        # blocks: (x0, y0, x1, y1, text, block_no, block_type, ...)
        blocks = [b for b in blocks if len(b) >= 5 and isinstance(b[4], str)]
        # Сортировка сверху-вниз, затем слева-направо
        blocks.sort(key=lambda b: (round(b[1], 2), round(b[0], 2)))
        lines: List[str] = []
        for b in blocks:
            t = (b[4] or "").strip()
            if t:
                lines.append(t)
        text = "\n".join(lines)
        # Если direct text пустой, пробуем простой режим
        if not text.strip():
            text = (page.get_text("text") or "").strip()
        return text
    except Exception:
        try:
            return (page.get_text("text") or "").strip()
        except Exception:
            return ""


def _extract_all_pages_text_via_ocr_only(pdf_bytes: bytes, dpi: int, max_pages: int) -> str:
    """Запасной путь: полностью через растеризацию + OCR (как было раньше)."""
    from src.pipeline.ocr import run_ocr
    import gc

    images = pdf_to_images(pdf_bytes, dpi=max(300, dpi), max_pages=max_pages)
    if not images:
        raise ValueError("PDF не содержит страниц или не удалось конвертировать")

    all_text = []
    for i, img in enumerate(images):
        try:
            print(f"🖼️ OCR only: страница {i+1}/{len(images)} (dpi={dpi})")
            page_text = run_ocr(img)
            if page_text.strip():
                all_text.append(f"=== СТРАНИЦА {i+1} ===\n{page_text}")
        except Exception as e:
            print(f"⚠️ Ошибка на странице {i+1}: {e}")
        finally:
            del img
            gc.collect()

    del images
    gc.collect()
    return "\n\n".join(all_text)


def _looks_corrupted_text(text: str) -> bool:
    """Эвристика: текст битый, если доминируют одинаковые символы (I, l, |)
    и низкое разнообразие букв/цифр. Часто бывает в PDF без ToUnicode для кириллицы.
    """
    if not text:
        return True
    s = text.replace("\n", "")
    if len(s) < 20:
        return True
    # Уникальность символов
    uniq_ratio = len(set(s)) / len(s)
    # Доля проблемных символов среди букв
    letters = [ch for ch in s if ch.isalpha()]
    if letters:
        bad = sum(1 for ch in letters if ch in {'I', 'l', '|', 'ı', 'İ'})
        bad_ratio = bad / len(letters)
    else:
        bad_ratio = 0.0
    # Доля кириллицы
    cyr = sum(1 for ch in s if 'А' <= ch <= 'я' or ch in 'Ёё')
    cyr_ratio = cyr / len(s)
    # Признаки битости: низкое разнообразие или очень высокая доля I/| среди букв
    if bad_ratio >= 0.5:
        return True
    if uniq_ratio < 0.20 and len(letters) > 0:
        return True
    # Если в тексте ожидается кириллица (есть ключевые слова) но cyr_ratio аномально мал
    keywords = ['КОНТРАКТ', 'Дата', 'Контрагент', 'Страна', 'Валюта', 'Сумма']
    if any(kw in text for kw in keywords) and cyr_ratio < 0.02:
        return True
    return False

def is_pdf_file(file_bytes: bytes) -> bool:
    """
    Проверяет, является ли файл PDF по magic bytes.
    
    Args:
        file_bytes: Первые байты файла
        
    Returns:
        True если файл PDF
    """
    return file_bytes.startswith(b'%PDF')