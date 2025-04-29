import streamlit as st
from utils import image_utils
from models import segmentation_2
import config
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.colors as mcolors
import random
import io
import base64

# Inicializace session state
if "masks" not in st.session_state:
    st.session_state.masks = None
if "labels" not in st.session_state:
    st.session_state.labels = None
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "show_original" not in st.session_state:
    st.session_state.show_original = True
if "custom_labels" not in st.session_state:
    st.session_state.custom_labels = {}
if "source" not in st.session_state:  # Nový řádek
    st.session_state.source = None

st.sidebar.title("SegmenStory")
st.sidebar.write("Aplikace pro segmentaci obrázků pomocí modelu SAM")

def mask_base64_to_array(mask_b64):
    try:
        img_data = base64.b64decode(mask_b64)
        mask_img = Image.open(io.BytesIO(img_data)).convert("L")
        mask = np.array(mask_img)
        return (mask > 127).astype(np.uint8)
    except Exception as e:
        st.error(f"Chyba konverze masky: {str(e)}")
        return None

def visualize_segmentation(image, masks, labels):
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image)
    else:
        pil_img = image
    result_img = pil_img.copy()
    draw = ImageDraw.Draw(result_img, 'RGBA')
    
    try:
        font = ImageFont.truetype("Arial", 12)
    except:
        font = ImageFont.load_default()
    
    colors = list(mcolors.TABLEAU_COLORS.values())
    if len(masks) > len(colors):
        colors += ['#{:06x}'.format(random.randint(0, 0xFFFFFF)) for _ in range(len(masks)-len(colors))]
    
    segments_info = []
    
    for i, mask in enumerate(masks):
        try:
            # Konverze masky
            if isinstance(mask, str):
                mask = mask_base64_to_array(mask)
                if mask is None or mask.sum() == 0:
                    continue
            
            # Zpracování masky
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0:
                continue
                
            # ... zbytek vizualizační logiky ...
            
        except Exception as e:
            st.error(f"Chyba v segmentu {i}: {str(e)}")
            continue
    
    return result_img, segments_info

uploaded_file = st.sidebar.file_uploader("Nahrajte obrázek", type=["jpg", "jpeg", "png"])
if uploaded_file:
    if st.session_state.show_original:
        st.image(uploaded_file, caption="Nahraný obrázek", use_container_width=True)
    
    if st.sidebar.button("Segmentovat"):
        with st.spinner("Probíhá segmentace..."):
            img = image_utils.process_image(uploaded_file)
            masks, labels = segmentation_2.segment_image(img, config.HF_API_TOKEN)
            
            # Debug výpisy
            st.write(f"Obdrženo masek: {len(masks)}")
            if masks:
                st.write("První maska:", masks[0][:50] + "..." if isinstance(masks[0], str) else masks[0].shape)
            
            st.session_state.masks = masks
            st.session_state.labels = labels
            st.session_state.processed_image = img
            st.session_state.show_original = False
            st.session_state.custom_labels = {}

if st.session_state.masks and len(st.session_state.masks) > 0:
    # Zobrazení obrázků vedle sebe
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Původní obrázek")
        st.image(st.session_state.processed_image, use_container_width=True)
    
    with col2:
        st.subheader("Segmentovaný obrázek")
        segmented_img, segments_info = visualize_segmentation(st.session_state.processed_image, st.session_state.masks, st.session_state.labels)
        st.image(segmented_img, use_container_width=True)
    
    # ... zbytek kódu ...

if st.sidebar.button("Nahrát nový obrázek"):
    # Reset všech stavů
    keys = list(st.session_state.keys())
    for key in keys:
        del st.session_state[key]
    st.rerun()
