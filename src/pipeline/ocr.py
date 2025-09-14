# src/pipeline/ocr.py
import numpy as np
from PIL import Image
import statistics

# Глобальная переменная для ленивой загрузки модели
_ocr_model = None

def _get_ocr_model():
    """Ленивая загрузка модели OCR"""
    global _ocr_model
    if _ocr_model is None:
        try:
            from paddleocr import PaddleOCR
            print("🤖 Загружаем модель PaddleOCR...")
            # Минимальная инициализация без дополнительных параметров
            _ocr_model = PaddleOCR(lang='ru')
            print("✅ Модель PaddleOCR загружена")
        except Exception as e:
            print(f"❌ Ошибка загрузки PaddleOCR: {e}")
            raise
    return _ocr_model

def _flatten_result(result):
    # PaddleOCR может вернуть [[(box,(txt,conf)),...]] или [(box,(txt,conf)),...]
    if not result:
        return []
    if isinstance(result[0], list):
        return result[0]
    return result

def _group_boxes_to_lines(items):
    """
    items: list of tuples (box, (text, conf))
    Возвращает список строк.
    """
    boxes = []
    for item in items:
        try:
            box = item[0]
            txt = item[1][0] if len(item) > 1 and isinstance(item[1], (list, tuple)) else str(item[1])
        except Exception:
            continue
        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        left = min(xs)
        right = max(xs)
        top = min(ys)
        bottom = max(ys)
        mid_y = (top + bottom) / 2
        height = bottom - top
        boxes.append({"text": txt, "left": left, "right": right, "mid_y": mid_y, "height": max(1, height)})

    if not boxes:
        return []

    # Сортируем по Y, потом по X
    boxes.sort(key=lambda b: (b["mid_y"], b["left"]))

    # Группируем по строкам
    heights = [b["height"] for b in boxes]
    median_h = statistics.median(heights) if heights else 10
    line_tol = max(8, median_h * 0.6)

    lines = []
    current = [boxes[0]]
    cur_y = boxes[0]["mid_y"]

    for b in boxes[1:]:
        if abs(b["mid_y"] - cur_y) <= line_tol:
            current.append(b)
            cur_y = (cur_y * (len(current) - 1) + b["mid_y"]) / len(current)
        else:
            lines.append(_merge_boxes_into_words(current, median_h))
            current = [b]
            cur_y = b["mid_y"]

    if current:
        lines.append(_merge_boxes_into_words(current, median_h))

    return lines


def _merge_boxes_into_words(boxes, median_h):
    """Объединяем символы в слова по горизонтали."""
    if not boxes:
        return ""
    
    boxes.sort(key=lambda x: x["left"])
    words = []
    current_word = boxes[0]["text"]

    for i in range(1, len(boxes)):
        gap = boxes[i]["left"] - boxes[i - 1]["right"]

        if gap > median_h * 0.7:  # если промежуток больше ~70% высоты, считаем разрыв слова
            words.append(current_word)
            current_word = boxes[i]["text"]
        else:
            current_word += boxes[i]["text"]

    words.append(current_word)
    return " ".join(words)

def run_ocr(pil_img: Image.Image, do_preprocess: bool = True) -> str:
    """Возвращает текст с объединением символов в строки."""
    try:
        # Получаем модель OCR (ленивая загрузка)
        ocr_model = _get_ocr_model()
        
        if do_preprocess:
            try:
                from src.pipeline.preprocess import preprocess_pil_image
                # Для больших страниц включаем более сильную обработку
                strong = pil_img.width * pil_img.height > 1800 * 1800
                img = preprocess_pil_image(pil_img, strong=strong)
            except Exception:
                img = pil_img.convert("RGB")
        else:
            img = pil_img.convert("RGB")

        # Уменьшаем размер изображения для экономии памяти
        max_size = 2200
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        arr = np.array(img)
        
        # Запускаем OCR без дополнительных параметров (они устарели)
        result = ocr_model.ocr(arr)
        
        if not result:
            return ""
        
        # Новый формат PaddleOCR - результат это список с одним элементом-словарем
        if isinstance(result, list) and len(result) > 0:
            ocr_data = result[0]
            
            # Извлекаем тексты и конфиденции из нового формата
            if isinstance(ocr_data, dict):
                rec_texts = ocr_data.get('rec_texts', [])
                rec_scores = ocr_data.get('rec_scores', [])
                
                # Фильтруем по уровню уверенности и собираем текст
                lines = []
                for i, text in enumerate(rec_texts):
                    confidence = rec_scores[i] if i < len(rec_scores) else 0.0
                    if confidence > 0.5:  # Порог уверенности
                        lines.append(str(text))
                
                result_text = "\n".join(lines)
                
                return result_text
            else:
                # Fallback для старого формата
                return _parse_old_format(result)
        else:
            return ""
        
    except Exception as e:
        print(f"ERROR в run_ocr: {e}")
        import traceback
        traceback.print_exc()
        return ""

def _parse_old_format(result):
    """Парсинг старого формата PaddleOCR"""
    try:
        items = _flatten_result(result)
        lines = _group_boxes_to_lines(items)
        
        if not lines:
            txts = []
            for it in items:
                try:
                    t = it[1][0] if len(it) > 1 else str(it[1])
                    txts.append(t)
                except Exception:
                    pass
            return " ".join(txts) if txts else ""
        
        return "\n".join(lines)
    except Exception as e:
        print(f"ERROR в _parse_old_format: {e}")
        return ""
