import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path
from utils.style_loader import load_css

from ocr.ocr_engine import (
    ocr_pdf,
    ocr_image,
    ocr_image_with_confidence,
)
from ocr.preprocessor import (
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

# ── CONFIG PAGE ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OCR Facture Automation",
    page_icon="🧾",
    layout="wide"
)

load_css()

st.markdown("""
<div class="main-header">
    <h1>OCR Facture Automation</h1>
    <p>Extraction automatique du texte · Python · OpenCV · Tesseract</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Options")
    st.markdown("---")
    mode   = st.radio("Type de fichier", ["Image", "PDF"])
    engine = st.radio(
        "Moteur OCR",
        ["Tesseract", "EasyOCR", "Compare tesseract et easyocr"]
    )
    st.markdown("---")
    show_confidence = st.checkbox("Score de confiance", value=False)
    save_result     = st.checkbox("Sauvegarder resultat", value=False)


# ── UTILITAIRE ────────────────────────────────────────────────────────────────
def afficher_etapes(steps: dict):
    cols = st.columns(len(steps))
    for col, (label, img) in zip(cols, steps.items()):
        with col:
            st.caption(label)
            st.image(img, use_container_width=True)


# ── MODE IMAGE ────────────────────────────────────────────────────────────────
if mode == "Image":

    uploaded = st.file_uploader(
        "Choisir une image de facture",
        type=["jpg", "jpeg", "png", "bmp", "tiff"]
    )

    if uploaded:
        if not is_valid_image(uploaded):
            st.error("Format non supporte.")
            st.stop()

        img_bgr = load_image_from_upload(uploaded)
        img_rgb = bgr_to_rgb(img_bgr)
        img_pil = Image.fromarray(img_rgb)

        st.markdown("#### Etapes de pretraitement")
        afficher_etapes(preprocess_image_steps(img_pil))
        st.markdown("---")

        if st.button("Lancer l'OCR", type="primary", use_container_width=True):
            with st.spinner("Extraction en cours..."):
                texte_final = ""  # valeur par defaut

                # ── Tesseract ─────────────────────────────────────────────
                if engine == "Tesseract":
                    if show_confidence:
                        result      = ocr_image_with_confidence(img_pil)
                        texte_final = result["text"]
                        col1, col2  = st.columns(2)
                        col1.metric("Confiance moyenne", f"{result['avg_confidence']}%")
                        col2.metric("Mots detectes", len(result["words"]))
                        st.text_area("Resultat OCR", texte_final, height=300)
                        with st.expander("Detail mot par mot"):
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
                        st.text_area("Resultat OCR", texte_final, height=350)

                # ── EasyOCR ───────────────────────────────────────────────
                elif engine == "EasyOCR":
                    from ocr.easyocr_engine import ocr_image_easyocr
                    result = ocr_image_easyocr(img_pil)
                    if "error" in result:
                        st.error(f"EasyOCR : {result['error']}")
                        texte_final = ""
                    else:
                        texte_final = result["text"]
                        st.metric("Confiance", f"{result['avg_confidence']}%")
                        st.text_area("Resultat", texte_final, height=300)

                # ── Comparaison ───────────────────────────────────────────
                elif engine == "Compare tesseract et easyocr":
                    from ocr.easyocr_engine import compare_tesseract_easyocr
                    results     = compare_tesseract_easyocr(img_pil)
                    texte_final = results["best"]["text"]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Tesseract")
                        st.metric("Confiance", f"{results['tesseract']['avg_confidence']}%")
                        st.text_area("Texte Tesseract", results["tesseract"]["text"],
                                     height=250, key="tess_text")
                    with col2:
                        st.markdown("#### EasyOCR")
                        st.metric("Confiance", f"{results['easyocr']['avg_confidence']}%")
                        st.text_area("Texte EasyOCR", results["easyocr"]["text"],
                                     height=250, key="easy_text")

                    best_name = results["best"]["engine"].capitalize()
                    st.success(f"Meilleur resultat : **{best_name}**")

                    if "error" in results["tesseract"]:
                        st.warning(f"Tesseract : {results['tesseract']['error']}")
                    if "error" in results["easyocr"]:
                        st.warning(f"EasyOCR : {results['easyocr']['error']}")

            st.success("Extraction terminee !")

            # ── Telechargement ────────────────────────────────────────────
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "Telecharger (.txt)",
                    data=texte_final,
                    file_name=f"{uploaded.name}_resultat.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            if save_result:
                with col_dl2:
                    path = save_output(f"{uploaded.name}_resultat.txt", texte_final)
                    st.info(f"Sauvegarde : {path}")


# ── MODE PDF ──────────────────────────────────────────────────────────────────
elif mode == "PDF":

    uploaded_files = st.file_uploader(
        "Deposer vos factures PDF",
        accept_multiple_files=True,
        type="pdf"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_key = uploaded_file.name

            with st.expander(
                f"{file_key}  —  {uploaded_file.size / 1024:.1f} KB",
                expanded=False
            ):
                if st.button("Lancer l'OCR", key=f"btn_{file_key}",
                             type="primary", use_container_width=True):
                    try:
                        pdf_path = save_uploaded_pdf(uploaded_file)

                        with st.spinner("Conversion PDF en images..."):
                            images = convert_from_path(str(pdf_path), dpi=300)

                        if not images:
                            st.error("Le PDF est vide !")
                            continue

                        st.markdown(
                            f'<span class="badge">{len(images)} page(s)</span>',
                            unsafe_allow_html=True
                        )
                        st.markdown("---")
                        st.markdown("### Apercu des pages")

                        for i, page_img in enumerate(images):
                            st.markdown(
                                f'<div class="page-card"><b>Page {i+1}</b></div>',
                                unsafe_allow_html=True
                            )
                            afficher_etapes(preprocess_image_steps(page_img))
                            st.divider()

                        extracted_text = ""

                        # ── Tesseract ─────────────────────────────────────
                        if engine == "Tesseract":
                            with st.spinner("Extraction Tesseract..."):
                                extracted_text = ocr_pdf(str(pdf_path))
                            st.success("OCR termine !")
                            st.markdown("### Texte extrait")
                            st.text_area("Resultat complet", extracted_text,
                                         height=400, key=f"text_{file_key}")

                        # ── EasyOCR ───────────────────────────────────────
                        elif engine == "EasyOCR":
                            from ocr.easyocr_engine import ocr_pdf_easyocr
                            with st.spinner("Extraction EasyOCR..."):
                                result = ocr_pdf_easyocr(str(pdf_path))
                            if "error" in result:
                                st.error(f"EasyOCR : {result['error']}")
                            else:
                                extracted_text = result["text"]
                                st.success("OCR termine !")
                                st.metric("Confiance moyenne", f"{result['avg_confidence']}%")
                                st.markdown("### Texte extrait")
                                st.text_area("Resultat complet", extracted_text,
                                             height=400, key=f"text_{file_key}")

                        # ── Comparaison ───────────────────────────────────
                        elif engine == "Compare tesseract et easyocr":
                            from ocr.easyocr_engine import compare_tesseract_easyocr_pdf
                            with st.spinner("Comparaison Tesseract vs EasyOCR..."):
                                results = compare_tesseract_easyocr_pdf(str(pdf_path))

                            extracted_text = results["best"]["text"]
                            st.success("Comparaison terminee !")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("#### Tesseract")
                                st.metric("Confiance", f"{results['tesseract']['avg_confidence']}%")
                                st.text_area("Texte Tesseract", results["tesseract"]["text"],
                                             height=300, key=f"tess_{file_key}")
                            with col2:
                                st.markdown("#### EasyOCR")
                                st.metric("Confiance", f"{results['easyocr']['avg_confidence']}%")
                                st.text_area("Texte EasyOCR", results["easyocr"]["text"],
                                             height=300, key=f"easy_{file_key}")

                            best_name = results["best"]["engine"].capitalize()
                            st.success(f"Meilleur resultat : **{best_name}**")

                            if "error" in results["tesseract"]:
                                st.warning(f"Tesseract : {results['tesseract']['error']}")
                            if "error" in results["easyocr"]:
                                st.warning(f"EasyOCR : {results['easyocr']['error']}")

                        # ── Telechargement ────────────────────────────────
                        if extracted_text:
                            st.download_button(
                                "Telecharger (.txt)",
                                data=extracted_text,
                                file_name=f"{file_key}_resultat.txt",
                                mime="text/plain",
                                key=f"dl_{file_key}",
                                use_container_width=True
                            )
                            if save_result:
                                path = save_output(f"{file_key}_resultat.txt", extracted_text)
                                st.info(f"Sauvegarde : {path}")

                    except Exception as e:
                        st.error(f"Erreur : {e}")
                        st.exception(e)