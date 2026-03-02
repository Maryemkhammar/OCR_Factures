import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
from PIL import Image

# Chemin Tesseract Windows ()
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# SUPPRESSION DES COULEURS (STYLO BLEU/ROUGE) 
def clean_color_ink_hybrid(image_color,
                           thr_blue=10,      # 15-25 (plus bas = plus de bleu clair)
                           thr_red=20,       # 20-45 (plus bas = plus permissif)
                           sat_min=20,       # 20-40 
                           val_min=25,       # protège du bruit sombre
                           dilate_iter=2,
                           inpaint_radius=2):

    b, g, r = cv2.split(image_color)
    max_rg = cv2.max(r, g)

    # Bleu robuste: bleu domine rouge ET vert
    mask_blue_rgb = cv2.threshold(cv2.subtract(b, max_rg), thr_blue, 255, cv2.THRESH_BINARY)[1]

    # Rouge plus souple: rouge domine un peu le bleu (moins strict)
    mask_red_rgb = cv2.threshold(cv2.subtract(r, b), thr_red, 255, cv2.THRESH_BINARY)[1]

    # HSV
    hsv = cv2.cvtColor(image_color, cv2.COLOR_BGR2HSV)

    # Filtre "pixels colorés" (évite texte noir)
    mask_hsv = cv2.inRange(hsv, (0, sat_min, val_min), (180, 255, 255))

    # IMPORTANT: capter le magenta/rose 
    # Hue magenta ~ [135..170] 
    mask_magenta = cv2.inRange(hsv, (125, sat_min, val_min), (180, 255, 255))

    # Combine
    base = cv2.bitwise_or(mask_blue_rgb, mask_red_rgb)
    base = cv2.bitwise_or(base, mask_magenta)

    final_mask = cv2.bitwise_and(base, mask_hsv)

    # Nettoyage masque
    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    final_mask = cv2.dilate(final_mask, kernel, iterations=dilate_iter)

    # Inpainting
    result = cv2.inpaint(image_color, final_mask, inpaint_radius, cv2.INPAINT_TELEA)
    return result

#___deskwing DÉTECTION ET CROP DE LA ZONE CONTENU
def detect_content_zone(image_color: np.ndarray) -> np.ndarray:

    gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # Binarisation robuste + inversion
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Connecter les blocs de texte/tableau
    kx = max(25, W // 80)
    ky = max(7, H // 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    connected = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    connected = cv2.morphologyEx(connected, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image_color  # fallback : retourne l'image entière

    best = None
    best_score = -1

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        if area < 0.08 * W * H:
            continue
        if w < 0.55 * W:
            continue
        if h < 0.35 * H:
            continue

        fill_ratio = cv2.contourArea(c) / (area + 1e-6)
        score = area * (0.5 + fill_ratio)

        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        c = max(contours, key=cv2.contourArea)
        best = cv2.boundingRect(c)

    x, y, w, h = best
    return image_color[y:y + h, x:x + w].copy()


#  PRÉTRAITEMENT DE L'IMAGE 
def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Pipeline OCR  :
    1. Conversion PIL → OpenCV
    2. Suppression stylo
    3. Gris
    4. Deskew
    5. Crop
    6. Débruitage
    7. Binarisation
    """

    # 1️ PIL → OpenCV BGR
    image_color = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

    # 2️ Suppression traces stylo (bleu / rouge/sature)
    image_color=clean_color_ink_hybrid(image_color)
    

    # 3️ Conversion en gris
    gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    # 4️ Deskew
    image_color_cropped=detect_content_zone(image_color)
    gray=cv2.cvtColor(image_color_cropped, cv2.COLOR_BGR2GRAY)
   
    # 5 Débruitage
    gray = cv2.medianBlur(gray, 3)

    # 7️ Binarisation finale
    _, final = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return final


#  OCR SUR UN PDF
def ocr_pdf(pdf_path: str) -> str:
    """OCR sur toutes les pages du PDF."""
    text_result = ""
    images = convert_from_path(pdf_path, dpi=300)

    for i, img in enumerate(images):
        processed = preprocess_image(img)

        text = pytesseract.image_to_string(
            processed,
            lang="fra",
            config="--psm 6"   # meilleur pour blocs de texte
        )
        text_result += f"\n--- Page {i+1} ---\n{text}"

    return text_result


# OCR SUR UNE IMAGE 
def ocr_image(img: Image.Image) -> str:
    """OCR sur une image PIL directement."""
    processed = preprocess_image(img)

    text = pytesseract.image_to_string(
        processed,
        lang="fra",
        config="--psm 6"
    )
    return text.strip()


# ─── OCR AVEC SCORE DE CONFIANCE ────────────────────────────
def ocr_image_with_confidence(img: Image.Image) -> dict:
    """
    OCR avec détail mot par mot + score de confiance.
    Retourne : { "text": ..., "avg_confidence": ..., "words": [...] }
    """
    processed = preprocess_image(img)

    data = pytesseract.image_to_data(
        processed,
        lang="fra",
        config="--psm 6",
        output_type=pytesseract.Output.DICT
    )

    words = []
    confidences = []

    for i in range(len(data["text"])):
        conf = int(data["conf"][i])
        word = data["text"][i].strip()

        if conf > 0 and word:
            words.append({"word": word, "confidence": conf})
            confidences.append(conf)

    avg_conf = round(np.mean(confidences), 1) if confidences else 0.0
    full_text = " ".join([w["word"] for w in words])

    return {
        "text": full_text,
        "avg_confidence": avg_conf,
        "words": words
    }

def preprocess_image_steps(img :Image.Image) -> dict:
    """chaque etapee du pretraitement pour affichage """
    
    steps={}
    
    #1 PIL -> bgr 
    image_color=cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    steps["1.Original"] = cv2.cvtColor(image_color , cv2.COLOR_BGR2RGB)
    #SUPPRESSION STYLO
    image_color= clean_color_ink_hybrid(image_color)
    steps["2. Supression stylooo"]= cv2.cvtColor(image_color , cv2.COLOR_BGR2RGB)
    
    #CROP
    
    image_color=detect_content_zone(image_color)
    steps["3.Crop intelligent"] = cv2.cvtColor(image_color , cv2.COLOR_BGR2RGB)
    
    # 4 gris + debruitage
    gray= cv2.cvtColor(image_color , cv2.COLOR_BGR2GRAY)
    gray=cv2.medianBlur(gray, 3)
    steps["4. gris + debruitage"]=gray
    
   # 5 binarisation finale
    _, final = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    steps["5. Binarisation"] = final

    return steps