import os
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
from pathlib import Path


# ─── DOSSIERS DU PROJET ──────────────────────────────────────
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── CHARGER UNE IMAGE UPLOADÉE (depuis Streamlit) ───────────
def load_image_from_upload(uploaded_file) -> np.ndarray:
    """
    Convertit un fichier uploadé Streamlit
    en tableau NumPy format OpenCV (BGR).
    """
    img_bytes = uploaded_file.read()
    img_pil   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np    = np.array(img_pil)
    img_bgr   = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_bgr


# ─── CONVERTIR BGR → RGB POUR AFFICHAGE STREAMLIT ───────────
def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """
    OpenCV = BGR, Streamlit = RGB.
    Toujours convertir avant st.image().
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ─── SAUVEGARDER UN RÉSULTAT ─────────────────────────────────
def save_output(filename: str, content: str) -> Path:
    """
    Sauvegarde le texte extrait dans le dossier outputs/.
    Retourne le chemin du fichier sauvegardé.
    """
    output_path = OUTPUT_DIR / filename
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path


# ─── VÉRIFIER LE FORMAT DU FICHIER ───────────────────────────
def is_valid_image(uploaded_file) -> bool:
    """Vérifie que le fichier est bien une image acceptée."""
    FORMATS_ACCEPTES = ["jpg", "jpeg", "png", "bmp", "tiff"]
    extension = uploaded_file.name.split(".")[-1].lower()
    return extension in FORMATS_ACCEPTES


def is_valid_pdf(uploaded_file) -> bool:
    """Vérifie que le fichier est bien un PDF."""
    return uploaded_file.name.lower().endswith(".pdf")


# ─── INFOS SUR L'IMAGE ───────────────────────────────────────
def get_image_info(img: np.ndarray) -> dict:
    """
    Retourne les informations de base d'une image.
    Utile pour l'affichage dans Streamlit.
    """
    h, w = img.shape[:2]
    canaux = img.shape[2] if len(img.shape) == 3 else 1

    return {
        "largeur":  w,
        "hauteur":  h,
        "canaux":   canaux,
        "taille_px": f"{w} × {h}",
        "type":     str(img.dtype)
    }


# ─── SAUVEGARDER UN PDF UPLOADÉ (depuis Streamlit) ───────────
def save_uploaded_pdf(uploaded_file) -> Path:
    """
    Sauvegarde un PDF uploadé via st.file_uploader()
    dans un fichier temporaire et retourne son chemin.
    """
    # Remettre le curseur au début
    uploaded_file.seek(0)

    # Lire le contenu
    data = uploaded_file.read()

    # Vérifier que le fichier n'est pas vide
    if not data:
        raise ValueError("Le fichier téléchargé est vide ou illisible.")

    # Créer un fichier temporaire .pdf sur le disque
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(data)
    tmp.close()

    return Path(tmp.name)