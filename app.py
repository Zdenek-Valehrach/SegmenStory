import streamlit as st
from utils import image_utils
from models import segmentation
import config
import numpy as np

# Inicializace proměnných v session state
if "labels" not in st.session_state:
    st.session_state.labels = []
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "show_original" not in st.session_state:
    st.session_state.show_original = True

# Sidebar
st.sidebar.title("SegmenStory")
st.sidebar.write("Aplikace pro segmentaci obrázků pomocí Mask2Former")

# Hlavní část aplikace
uploaded_file = st.sidebar.file_uploader("Nahrajte obrázek", type=["jpg", "jpeg", "png"])

if uploaded_file:
    if st.session_state.show_original:
        st.image(uploaded_file, caption="Nahraný obrázek", use_container_width=True)

    if st.sidebar.button("Segmentovat"):
        with st.spinner("Probíhá segmentace obrázku..."):
            img = image_utils.process_image(uploaded_file)
            _, labels = segmentation.segment_image(img, config.HF_API_TOKEN)
            st.session_state.labels = list(set(labels))  # Deduplikace tříd
            st.session_state.processed_image = img
            st.session_state.show_original = False
            st.rerun()

# Zobrazení výsledků
if st.session_state.labels:
    unique_labels = sorted(st.session_state.labels)
    st.subheader("Nalezené třídy objektů v obrázku:")
    selected_class = st.selectbox(
        "Vyberte třídu:",
        unique_labels,
        index=0,
        key="main_label_selectbox"
    )

# Tlačítko pro resetování
if st.sidebar.button("Nová segmentace"):
    st.session_state.labels = []
    st.session_state.processed_image = None
    st.session_state.show_original = True
    st.rerun()
