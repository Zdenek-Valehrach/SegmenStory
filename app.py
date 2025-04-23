import streamlit as st
from utils import image_utils, prompt_utils
from models import segmentation, llm
import config

st.title("SegmenStory")
st.write("Aplikace pro segmentaci obrázků a generování textů")

uploaded_file = st.file_uploader("Nahrajte obrázek", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Nahraný obrázek", use_column_width=True)
    
    if st.button("Segmentovat"):
        with st.spinner("Probíhá segmentace obrázku..."):
            img = image_utils.process_image(uploaded_file)
            masks, labels = segmentation.segment_image(img, config.HF_API_TOKEN)
            
            st.success(f"Segmentace dokončena! Nalezeno {len(masks)} objektů.")
            # TODO: Implementovat zobrazení segmentovaných objektů