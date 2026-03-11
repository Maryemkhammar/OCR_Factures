import pytesseract
from pdf2image import convert_from_path
import numpy as np
from PIL import Image

from ocr.preprocessor import preprocess_image

# Chemin Tesseract Windows ()
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#  OCR SUR UN PDF
def ocr_pdf(pdf_path: str) -> str:
    """OCR sur toutes les pages du PDF."""
    try:
        images = convert_from_path(pdf_path, dpi=300)
    except Exception as e :
        return f"[ERREUR] Impossible de lire le PDF :{e}"
        
    text_result = ""
    for i, img in enumerate(images):
        try:
            processed   = preprocess_image(img)
            text        = pytesseract.image_to_string(
                processed, lang="fra", config="--psm 3"
            )
            text_result += f"\n--- Page {i+1} ---\n{text}"
        except Exception as e:
            text_result += f"\n--- Page {i+1} --- [ERREUR : {e}]\n"

    return text_result
      


# OCR SUR UNE IMAGE 
def ocr_image(img: Image.Image) -> str:
    try:
        processed = preprocess_image(img)
        return pytesseract.image_to_string(
            processed,
            lang="fra",
            config="--psm 6"
        ).strip()
    except Exception as e:
        return f"[ERREUR] OCR échoué : {e}"



# ─── OCR AVEC SCORE DE CONFIANCE ────────────────────────────
def ocr_image_with_confidence(img: Image.Image) -> dict:
    try:
        processed = preprocess_image(img)
        data      = pytesseract.image_to_data(
            processed, lang="fra", config="--psm 3",
            output_type=pytesseract.Output.DICT
        )
    except Exception as e:
        return {"text": "", "avg_confidence": 0.0, "words": [], "error": str(e)}

    words, confidences = [], []
    for i in range(len(data["text"])):
        conf = int(data["conf"][i])
        word = data["text"][i].strip()
        if conf > 0 and word:
            words.append({"word": word, "confidence": conf})
            confidences.append(conf)

    avg_conf  = round(np.mean(confidences), 1) if confidences else 0.0
    full_text = " ".join([w["word"] for w in words])

    return {"text": full_text, "avg_confidence": avg_conf, "words": words}