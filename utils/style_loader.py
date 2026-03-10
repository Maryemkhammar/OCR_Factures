from pathlib import Path
import streamlit as st 

def load_css(css_file: str = "styles/main.css") -> None:
    css_path=Path(css_file)
    if not css_path.exists():
        st.warning(f"css introuvable : {css_file}")
        return 
    css=css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)