import cv2
import numpy as np
import re
import os
 
try:
    from paddleocr import PaddleOCR
    _ocr = PaddleOCR(use_angle_cls=True, lang="en", use_GPU=False)
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("[ocr] PaddleOCR not available – OCR path disabled.")
 
#pre-processing
def preprocess_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    # Slight sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
 
    # Adaptive threshold works better than global threshold on varying backgrounds
    thresh = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=10
    )
    denoised = cv2.fastNlMeansDenoising(thresh, h=10)
    return denoised

def _write_preprocessed(image_path: str) -> str:
    processed = preprocess_image(image_path)
    base, ext = os.path.splitext(image_path)
    tmp_path = base + "_preprocessed.png"
    cv2.imwrite(tmp_path, processed)
    return tmp_path

#extract text
def extract_text_from_image(image_path: str) -> str:
    if not PADDLE_AVAILABLE:
        return ""
    tmp_path = None
    try:
        tmp_path = _write_preprocessed(image_path)
        result = _ocr.ocr(tmp_path, cls=True)
    except Exception as e:
        print(f"[ocr] extract_text_from_image failed on {image_path}: {e}")
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not result or not result[0]:
        return ""
 
    lines = []
    for line in result[0]:
        text = line[1][0]
        lines.append(text)
    return "\n".join(lines)

#construct table
def extract_table_rows_from_image(image_path: str, y_band: int = 15) -> list:
    if not PADDLE_AVAILABLE:
        return []
    tmp_path = None
    
    try:
        tmp_path = _write_preprocessed(image_path)
        result = _ocr.ocr(tmp_path, cls=True)
    except Exception as e:
        print(f"[ocr] extract_table_rows_from_image failed on {image_path}: {e}")
        return []
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
 
    if not result or not result[0]:
        return []

    # Each item: (y_center, x_center, text)
    items = []
    for line in result[0]:
        box = line[0]          # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        text = line[1][0]
        x_center = (box[0][0] + box[2][0]) / 2
        y_center = (box[0][1] + box[2][1]) / 2
        items.append((y_center, x_center, text))
    
    if not items:
        return []
 
    # Sort by Y then X
    items.sort(key=lambda t: (t[0], t[1]))
 
    # Group by Y-band
    rows = []
    current_row = []
    current_y = items[0][0]
 
    for y, x, text in items:
        if abs(y - current_y) <= y_band:
            current_row.append((x, text))
        else:
            rows.append(sorted(current_row, key=lambda t: t[0]))
            current_row = [(x, text)]
            current_y = y
 
    if current_row:
        rows.append(sorted(current_row, key=lambda t: t[0]))
 
    # Strip x-coordinates, return only text
    return [[cell[1] for cell in row] for row in rows]
 
#filtering rows 
PATTERN = re.compile(r"^\d{3,6}$")
 
def filter_register_rows(rows: list) -> list:
    result = []
    for row in rows:
        for cell in row:
            if PATTERN.match(cell.strip()):
                result.append(row)
                break
    return result