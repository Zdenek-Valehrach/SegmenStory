import streamlit as st
from utils import image_utils
from models import segmentation
import config
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.colors as mcolors
import random
import io

# Inicializace proměnných v session state
if "masks" not in st.session_state:
    st.session_state.masks = None
if "labels" not in st.session_state:
    st.session_state.labels = None
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "show_original" not in st.session_state:
    st.session_state.show_original = True

# Přesouvám titulek a popis do postranního panelu
st.sidebar.title("SegmenStory")
st.sidebar.write("Aplikace pro segmentaci obrázků pomocí pokročilých AI modelů")

# Funkce pro vizualizaci segmentovaných oblastí
def visualize_segmentation(image, masks, labels):
    # Konverze numpy array na PIL Image
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image)
    else:
        pil_img = image
    
    # Vytvoření kopie pro kreslení
    result_img = pil_img.copy()
    draw = ImageDraw.Draw(result_img, 'RGBA')
    
    # Pokus o načtení fontu
    try:
        font_size = 12
        font = ImageFont.truetype("Arial", font_size)
    except IOError:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Generování různých barev pro segmenty
    colors = list(mcolors.TABLEAU_COLORS.values())
    if len(masks) > len(colors):
        additional_colors = ['#{:06x}'.format(random.randint(0, 0xFFFFFF)) for _ in range(len(masks) - len(colors))]
        colors.extend(additional_colors)
    
    # Seznam pro uložení informací o segmentech
    segments_info = []
    
    # Vykreslení segmentů
    for i, mask in enumerate(masks):
        # Přeměna masky na numpy array (pokud již není)
        if not isinstance(mask, np.ndarray):
            try:
                if isinstance(mask, str):
                    import base64
                    img_data = base64.b64decode(mask)
                    mask_img = Image.open(io.BytesIO(img_data))
                    mask = np.array(mask_img)
                else:
                    mask = np.array(mask)
            except Exception as e:
                continue
        
        # Ověříme platnost masky
        if mask.size == 0:
            continue
        
        # Barva pro tento segment
        color = colors[i % len(colors)]
        hex_color = color if isinstance(color, str) else mcolors.to_hex(color)
        
        # Převedeme na RGBA pro průhlednost a na RGB pro okraje
        if isinstance(color, str):
            if color.startswith('#'):
                color = color[1:]
            r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
            rgba_color = (r, g, b, 64)  # 64 je hodnota alpha (průhlednosti)
            rgb_color = (r, g, b)  # Plná barva pro ohraničení
        else:
            r, g, b = [int(x*255) for x in mcolors.to_rgb(color)]
            rgba_color = (r, g, b, 64)
            rgb_color = (r, g, b)
        
        # Získáme label pro tento objekt
        label = labels[i] if i < len(labels) else f"Objekt {i+1}"
        
        # Určení bounding boxu pro masku
        try:
            # Pro explicitní bounding box
            if isinstance(mask, list) and len(mask) == 4:
                x_min, y_min, width, height = mask
                x_max = x_min + width
                y_max = y_min + height
            # Pro masku v podobě pole
            else:
                # Najděme souřadnice aktivních pixelů v masce
                y_indices, x_indices = np.where(mask)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    width = x_max - x_min
                    height = y_max - y_min
                else:
                    continue  # Prázdná maska
            
            # Přidáme informaci o segmentu do seznamu
            segments_info.append({
                "id": i,
                "label": label,
                "coords": [x_min, y_min, width, height],
                "color": hex_color
            })
            
            # Vykreslení průhledné výplně masky
            if isinstance(mask, list) and len(mask) == 4:
                draw.rectangle([x_min, y_min, x_max, y_max], fill=(rgba_color))
            else:
                for y in range(pil_img.height):
                    for x in range(pil_img.width):
                        try:
                            if mask[y, x]:  # Pokud je pixel součástí masky
                                draw.point((x, y), fill=rgba_color)
                        except IndexError:
                            continue
            
            # Vykreslení ohraničujícího obdélníku
            outline_width = 2
            for offset in range(outline_width):
                draw.rectangle(
                    [x_min-offset, y_min-offset, x_max+offset, y_max+offset], 
                    outline=rgb_color,
                    width=1
                )
            
            # Vykreslení textu s popiskem objektu
            text_position = (x_min + 5, y_min + 5)
            
            # Zjistíme velikost textu pro pozadí
            if font:
                text_width, text_height = draw.textsize(label, font=font) if hasattr(draw, 'textsize') else (len(label) * 6, 12)
            else:
                text_width, text_height = len(label) * 6, 12
            
            # Vykreslení tmavého pozadí pod textem
            padding = 2
            background_box = [
                text_position[0] - padding, 
                text_position[1] - padding,
                text_position[0] + text_width + padding,
                text_position[1] + text_height + padding
            ]
            draw.rectangle(background_box, fill=(0, 0, 0, 180))
            
            # Vykreslení textu
            if font:
                draw.text(text_position, label, fill=(255, 255, 255), font=font)
            else:
                draw.text(text_position, label, fill=(255, 255, 255))
                
        except Exception as e:
            continue
    
    return result_img, segments_info

# Hlavní část aplikace - přesunuta do sidebar
uploaded_file = st.sidebar.file_uploader("Nahrajte obrázek", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Zobrazíme nahraný obrázek před segmentací v hlavním obsahu
    if st.session_state.show_original:
        st.image(uploaded_file, caption="Nahraný obrázek", use_container_width=True)
    
    # Tlačítko pro segmentaci v sidebaru
    if st.sidebar.button("Segmentovat"):
        with st.spinner("Probíhá segmentace obrázku..."):
            img = image_utils.process_image(uploaded_file)
            masks, labels = segmentation.segment_image(img, config.HF_API_TOKEN)
            
            # Uložíme výsledky do session state
            st.session_state.masks = masks
            st.session_state.labels = labels
            st.session_state.processed_image = img
            st.session_state.show_original = False
            
            st.rerun()
    
    # Zpracování segmentovaného obrázku, pokud existuje
    if st.session_state.masks and len(st.session_state.masks) > 0:
        masks = st.session_state.masks
        labels = st.session_state.labels
        img = st.session_state.processed_image
        
        # Vizualizace segmentace
        segmented_img, segments_info = visualize_segmentation(img, masks, labels)
        
        # Zobrazení segmentovaného obrázku v hlavním obsahu
        st.subheader("Výsledek segmentace")
        st.image(segmented_img, use_container_width=True)
        
        # Přesuneme rolovací nabídku pro výběr segmentů do sidebar
        segment_options = sorted(list(set([seg['label'] for seg in segments_info])))
        if segment_options:
            st.sidebar.selectbox("Seznam nalezených segmentů:", segment_options)
        
        # Přidáme tlačítko pro návrat k nahrání nového obrázku do sidebar
        if st.sidebar.button("Nahrát nový obrázek"):
            st.session_state.masks = None
            st.session_state.labels = None
            st.session_state.processed_image = None
            st.session_state.show_original = True
            st.rerun()