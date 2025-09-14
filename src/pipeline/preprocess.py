# src/pipeline/preprocess.py
import cv2
import numpy as np
from PIL import Image

def deskew_image(gray):
    coords = np.column_stack(np.where(gray < 255))
    if coords.shape[0] < 10:
        return gray
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_pil_image(pil_img: Image.Image, target_width:int=1800, strong: bool = False) -> Image.Image:
    """Подготовка изображения для OCR.
    strong=True включает усиление контраста (CLAHE) и лёгкое повышение резкости.
    """
    img = np.array(pil_img.convert("L"))

    # Масштабируем, если слишком маленькое
    h, w = img.shape[:2]
    if w < target_width:
        scale = target_width / w
        img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Дескью
    img = deskew_image(img)

    # Немного сглаживаем шум
    img = cv2.medianBlur(img, 3)

    if strong:
        # Усиливаем контраст локально (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        # Лёгкая резкость (unsharp mask)
        blur = cv2.GaussianBlur(img, (0,0), 1.0)
        img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    # Конвертируем обратно в RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)
