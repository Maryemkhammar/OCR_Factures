import streamlit as st
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

from ocr.ocr_engine import (
    ocr_pdf,
    ocr_image,
    ocr_image_with_confidence,
    preprocess_image,
    preprocess_image_steps
)
from utils.file_manager import (
    load_image_from_upload,
    bgr_to_rgb,
    save_output,
    is_valid_image,
    get_image_info,
    save_uploaded_pdf
)


#  CONFIG PAGE

st.set_page_config(
    page_title="OCR Facture Automation",
    page_icon="🧾",
    layout="wide"
)


#  CSS

st.markdown("""
<style>
    .stApp { background-color: #F8FAFC; }

    .main-header {
        background: linear-gradient(135deg, #0D1B2A 0%, #1B4F72 100%);
        padding: 22px 30px;
        border-radius: 14px;
        margin-bottom: 25px;
        border-left: 6px solid #06B6D4;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.9rem; }
    .main-header p  { color: #94D2E6; margin: 6px 0 0 0; font-size: 0.95rem; }

    .page-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .badge {
        background: #DBEAFE;
        color: #1E40AF;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .stButton > button { border-radius: 8px; font-weight: 600; }

    section[data-testid="stSidebar"] { background: #0D1B2A !important; }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] .stRadio label { color: #94D2E6 !important; }
</style>
""", unsafe_allow_html=True)


#  EN-TÊTE

st.markdown("""
<div class="main-header">
    <h1>🧾 OCR Facture Automation</h1>
    <p>Extraction automatique du texte · Python · OpenCV · Tesseract</p>
</div>
""", unsafe_allow_html=True)


#  SIDEBAR

with st.sidebar:
    st.markdown("## ⚙️ Options")
    st.markdown("---")

    mode = st.radio("Type de fichier", ["🖼️ Image", "📄 PDF"])

    st.markdown("---")

    show_confidence = st.checkbox("📊 Score de confiance", value=False)
    save_result     = st.checkbox("💾 Sauvegarder résultat", value=False)


#  FONCTION UTILITAIRE : affichage des étapes

def afficher_etapes(steps: dict, selectbox_key: str):
    """
    Affiche les miniatures des étapes de prétraitement
    + un selectbox pour agrandir une étape dans la page.
    """
    # Miniatures côte à côte
    cols = st.columns(len(steps))
    for col, (label, img) in zip(cols, steps.items()):
        with col:
            st.caption(label)
            st.image(img, use_container_width=True)

    # Agrandissement dans la page
    st.markdown("&nbsp;")
    choix = st.selectbox("🔍 Agrandir une étape", list(steps.keys()), key=selectbox_key)
    st.image(steps[choix], use_container_width=True)



#  MODE IMAGE

if mode == "🖼️ Image":

    uploaded = st.file_uploader(
        "📤 Choisir une image de facture",
        type=["jpg", "jpeg", "png", "bmp", "tiff"]
    )

    if uploaded:

        # Validation
        if not is_valid_image(uploaded):
            st.error("❌ Format non supporté.")
            st.stop()

        # Chargement
        img_bgr = load_image_from_upload(uploaded)
        img_rgb = bgr_to_rgb(img_bgr)
        img_pil = Image.fromarray(img_rgb)

        # Étapes de prétraitement
        st.markdown("#### 🖼️ Étapes de prétraitement")
        steps = preprocess_image_steps(img_pil)
        afficher_etapes(steps, selectbox_key="select_image")

        st.markdown("---")

        # Lancer OCR
        if st.button("▶️ Lancer l'OCR", type="primary", use_container_width=True):
            with st.spinner("🔍 Extraction en cours..."):

                if show_confidence:
                    result      = ocr_image_with_confidence(img_pil)
                    texte_final = result["text"]

                    col1, col2 = st.columns(2)
                    col1.metric("📊 Confiance moyenne", f"{result['avg_confidence']}%")
                    col2.metric("📝 Mots détectés", len(result["words"]))

                    st.text_area("📝 Résultat OCR", texte_final, height=300)

                    with st.expander("📊 Détail mot par mot"):
                        import pandas as pd
                        df = pd.DataFrame(result["words"])
                        st.dataframe(
                            df.style.background_gradient(
                                subset=["confidence"],
                                cmap="RdYlGn", vmin=0, vmax=100
                            ),
                            use_container_width=True
                        )
                else:
                    texte_final = ocr_image(img_pil)
                    st.text_area("📝 Résultat OCR", texte_final, height=350)

            st.success("✅ Extraction terminée !")

            # Téléchargement
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "💾 Télécharger (.txt)",
                    data=texte_final,
                    file_name=f"{uploaded.name}_resultat.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            if save_result:
                with col_dl2:
                    path = save_output(f"{uploaded.name}_resultat.txt", texte_final)
                    st.info(f"💾 Sauvegardé : {path}")


#  MODE PDF

elif mode == "📄 PDF":

    uploaded_files = st.file_uploader(
        "📤 Déposer vos factures PDF",
        accept_multiple_files=True,
        type="pdf"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_key = uploaded_file.name

            with st.expander(f"📄 {file_key}  —  {uploaded_file.size / 1024:.1f} KB", expanded=False):

                if st.button("▶️ Lancer l'OCR", key=f"btn_{file_key}", type="primary", use_container_width=True):
                    try:
                        # 1. Sauvegarde PDF temporaire
                        pdf_path = save_uploaded_pdf(uploaded_file)

                        # 2. Conversion PDF → images
                        with st.spinner("📄 Conversion PDF → images..."):
                            images = convert_from_path(str(pdf_path), dpi=300)

                        if not images:
                            st.error("❌ Le PDF est vide !")
                            continue

                        st.markdown(
                            f'<span class="badge">📑 {len(images)} page(s) détectée(s)</span>',
                            unsafe_allow_html=True
                        )
                        st.markdown("---")

                        # 3. Aperçu page par page avec étapes
                        st.markdown("### 🖼️ Aperçu des pages")

                        for i, page_img in enumerate(images):

                            st.markdown(
                                f'<div class="page-card"><b>Page {i+1}</b></div>',
                                unsafe_allow_html=True
                            )

                            steps = preprocess_image_steps(page_img)
                            afficher_etapes(steps, selectbox_key=f"select_p{i}_{file_key}")

                            st.divider()

                        # 4. OCR toutes les pages
                        with st.spinner("🔍 Extraction du texte..."):
                            extracted_text = ocr_pdf(str(pdf_path))

                        st.success("✅ OCR terminé !")

                        # 5. Résultat texte
                        st.markdown("### 📝 Texte extrait")
                        st.text_area(
                            "Résultat complet",
                            extracted_text,
                            height=400,
                            key=f"text_{file_key}"
                        )

                        # 6. Téléchargement
                        st.download_button(
                            "💾 Télécharger le texte (.txt)",
                            data=extracted_text,
                            file_name=f"{file_key}_resultat.txt",
                            mime="text/plain",
                            key=f"dl_{file_key}",
                            use_container_width=True
                        )

                        if save_result:
                            path = save_output(f"{file_key}_resultat.txt", extracted_text)
                            st.info(f"💾 Sauvegardé : {path}")

                    except Exception as e:
                        st.error(f"❌ Erreur : {e}")
                        st.exception(e)