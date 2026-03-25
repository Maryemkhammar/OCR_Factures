"""
ocr/easyocr_engine.py
"""
import threading
import numpy as np
from PIL import Image
from config.ocr_config import config

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

_reader = None
_lock   = threading.Lock()


def get_reader():
    global _reader
    with _lock:
        if _reader is None:
            if not EASYOCR_AVAILABLE:
                raise ImportError("EasyOCR non installe : pip install easyocr")
            if not config.easyocr.enabled:
                raise RuntimeError("EasyOCR desactive dans config/ocr_config.py")
            print("Chargement modele EasyOCR...")
            _reader = easyocr.Reader(
                config.easyocr.languages,
                gpu=config.easyocr.gpu,
                verbose=config.easyocr.verbose
            )
            print("Modele EasyOCR charge.")
    return _reader


def ocr_image_easyocr(img: Image.Image, preprocess: bool = False) -> dict:
    try:
        if preprocess:
            from ocr.preprocessor import preprocess_image
            arr = preprocess_image(img)
        else:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            arr = np.array(img)

        reader  = get_reader()
        results = reader.readtext(arr, detail=1)

        words = []
        for (bbox, text, conf) in results:
            words.append({
                "word":       text,
                "confidence": round(conf * 100, 1),
                "bbox":       bbox
            })

        avg_conf  = round(
            sum(w["confidence"] for w in words) / len(words), 1
        ) if words else 0.0

        full_text = " ".join(w["word"] for w in words)

        return {
            "engine":         "easyocr",
            "text":           full_text,
            "avg_confidence": avg_conf,
            "words":          words,
            "empty":          len(words) == 0,
        }

    except Exception as e:
        return {
            "engine":         "easyocr",
            "text":           "",
            "avg_confidence": 0.0,
            "words":          [],
            "empty":          False,
            "error":          str(e)
        }


def ocr_pdf_easyocr(pdf_path: str) -> dict:
    from pdf2image import convert_from_path
    import cv2
    from ocr.preprocessor import clean_color_ink_hybrid, deskew_image

    try:
        images         = convert_from_path(pdf_path, dpi=300)
        all_words      = []
        all_text_parts = []

        for img in images:
            image_color = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
            image_color = clean_color_ink_hybrid(image_color)
            image_color = deskew_image(image_color)
            img_clean   = Image.fromarray(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))

            result = ocr_image_easyocr(img_clean, preprocess=False)
            if "error" not in result:
                all_words.extend(result["words"])
                all_text_parts.append(result["text"])

        full_text = "\n".join(all_text_parts)
        avg_conf  = round(
            sum(w["confidence"] for w in all_words) / len(all_words), 1
        ) if all_words else 0.0

        return {
            "engine":         "easyocr",
            "text":           full_text,
            "avg_confidence": avg_conf,
            "words":          all_words,
            "empty":          len(all_words) == 0,
        }

    except Exception as e:
        return {
            "engine":         "easyocr",
            "text":           "",
            "avg_confidence": 0.0,
            "words":          [],
            "empty":          False,
            "error":          str(e)
        }


def compare_tesseract_easyocr(img: Image.Image) -> dict:
    # Import ici pour eviter l'import circulaire
    from ocr.ocr_engine import ocr_image_with_confidence

    tess           = ocr_image_with_confidence(img)
    tess["engine"] = "tesseract"
    easy           = ocr_image_easyocr(img, preprocess=False)

    tess_ok = "error" not in tess
    easy_ok = "error" not in easy

    if not tess_ok and not easy_ok:
        best = tess
    elif not easy_ok:
        best = tess
    elif not tess_ok:
        best = easy
    elif tess["avg_confidence"] >= easy["avg_confidence"]:
        best = tess
    else:
        best = easy

    return {"tesseract": tess, "easyocr": easy, "best": best}


def compare_tesseract_easyocr_pdf(pdf_path: str) -> dict:
    from ocr.ocr_engine import ocr_pdf

    try:
        tess_text = ocr_pdf(pdf_path)
        tess = {
            "engine":         "tesseract",
            "text":           tess_text,
            "avg_confidence": 0.0,
        }
    except Exception as e:
        tess = {
            "engine":         "tesseract",
            "text":           "",
            "avg_confidence": 0.0,
            "error":          str(e)
        }

    easy = ocr_pdf_easyocr(pdf_path)

    tess_ok = "error" not in tess
    easy_ok = "error" not in easy

    if not tess_ok and not easy_ok:
        best = tess
    elif not easy_ok:
        best = tess
    elif not tess_ok:
        best = easy
    elif easy["avg_confidence"] > 0:
        best = easy
    else:
        best = tess

    return {"tesseract": tess, "easyocr": easy, "best": best}