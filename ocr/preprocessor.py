import cv2
import numpy as np
from PIL import Image
# SUPPRESSION DES COULEURS (STYLO BLEU/ROUGE) 
def clean_color_ink_hybrid(image_color,
                           thr_blue=10,      # 15-25 (plus bas = plus de bleu clair)
                           thr_red=37,       # 20-45 (plus bas = plus permissif)
                           sat_min=30,       # 20-40 
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

    #  le magenta/rose 
    # Hue magenta  [135..170] 
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

#DESKEW 

def deskew_image(image_color:np.ndarray) ->np.ndarray:
    gray=cv2.cvtColor(image_color,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray, 50 , 150 , apertureSize=3)
    lines=cv2.HoughLinesP(edges, 1 , np.pi/180 ,
                          threshold=100,
                          minLineLength=100,
                          maxLineGap=10)
    if lines is None:
        return image_color
    
    angles=[]
    for x1,y1,x2,y2 in lines[:, 0]:
        if abs(x2-x1) > 0:
            angle=np.degrees(np.arctan2(y2 - y1 , x2 -x1))
            if -45 < angle < 45:
                angles.append(angle)
    if not angles:
        return image_color
    
    angle=np.median(angles)
    if abs(angle) < 0.5:
        return image_color
    h,w=image_color.shape[:2]
    M  = cv2.getRotationMatrix2D((w // 2 , h //2), angle, 1.0)
    return cv2.warpAffine(image_color,M,(w,h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

#__DÉTECTION ET CROP DE LA ZONE CONTENU
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
        return image_color  

    best = None
    best_score = -1

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        if area < 0.05 * W * H:
            continue
        if w < 0.35 * W:
            continue
        if h < 0.20 * H:
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
    3.Deskew
    4. Crop zone contenu
    5. gris + Débruitage
    6. Binarisation
    """

    # 1️ PIL → OpenCV BGR
    image_color = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

    # 2️ Suppression traces stylo (bleu / rouge/sature)
    image_color=clean_color_ink_hybrid(image_color)
    
    #3. deskew 
    image_color=deskew_image(image_color)
    
    #4. crop
    image_color=detect_content_zone(image_color)
    

    # 5. Conversion en gris
    gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    
   
    # 5 gris+ Débruitage
    gray=cv2.cvtColor(image_color,cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    # 7️ Binarisation finale
    _, final = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return final

def preprocess_image_steps(img :Image.Image) -> dict:
    """chaque etapee du pretraitement pour affichage """
    
    steps={}
    
    #1 PIL -> bgr 
    image_color=cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    steps["1.Original"] = cv2.cvtColor(image_color , cv2.COLOR_BGR2RGB)
    
    #2 SUPPRESSION STYLO
    image_color= clean_color_ink_hybrid(image_color)
    steps["2. Supression stylo"]= cv2.cvtColor(image_color , cv2.COLOR_BGR2RGB)
    
    #3 deskew 
    image_color=deskew_image(image_color)
    steps["3.Deskew"] = cv2.cvtColor(image_color , cv2.COLOR_BGR2RGB)
    
    #4 crop zone 
    image_color=detect_content_zone(image_color)
    steps["4. crop zone contenu"] = cv2.cvtColor(image_color , cv2.COLOR_BGR2RGB)
    
    
    # 4 gris + debruitage
    gray= cv2.cvtColor(image_color , cv2.COLOR_BGR2GRAY)
    gray=cv2.medianBlur(gray, 3)
    steps["5. gris + debruitage"]=gray
    
   # 5 binarisation finale
    _, final = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    steps["6. Binarisation"] = final

    return steps