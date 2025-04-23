import streamlit as st
from utils import image_utils, prompt_utils
from models import segmentation, llm
import config
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import random
import io

st.title("SegmenStory")
st.write("Aplikace pro segmentaci obrázků a generování textů")

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
    
    # Generování různých barev pro segmenty
    colors = list(mcolors.TABLEAU_COLORS.values())
    if len(masks) > len(colors):
        # Pokud je více masek než barev, generujeme náhodné barvy
        additional_colors = ['#{:06x}'.format(random.randint(0, 0xFFFFFF)) for _ in range(len(masks) - len(colors))]
        colors.extend(additional_colors)
    
    # Seznam pro uložení barev použitých pro každý objekt (pro legendu)
    object_colors = []
    
    # Vykreslení segmentů
    for i, mask in enumerate(masks):
        # Přeměna masky na numpy array (pokud již není)
        if not isinstance(mask, np.ndarray):
            try:
                # Pokud je maska string, převedeme ji na numerická data
                if isinstance(mask, str):
                    import base64
                    img_data = base64.b64decode(mask)
                    mask_img = Image.open(io.BytesIO(img_data))
                    mask = np.array(mask_img)
                # Jinak zkusíme převést na numpy array
                else:
                    mask = np.array(mask)
            except Exception as e:
                st.warning(f"Nelze zpracovat masku {i}: {str(e)}")
                continue
        
        # Ověříme platnost masky
        if mask.size == 0:
            continue
        
        # Barva pro tento segment
        color = colors[i % len(colors)]
        # Přidáme do seznamu barev pro legendu
        hex_color = color if isinstance(color, str) else mcolors.to_hex(color)
        object_colors.append(hex_color)
        
        # Převedeme na RGBA pro průhlednost
        if isinstance(color, str):
            # Převod hex barvy na RGBA
            if color.startswith('#'):
                color = color[1:]
            r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
            rgba_color = (r, g, b, 64)  # 64 je hodnota alpha (průhlednosti)
        else:
            # Převod matplotlib barvy na RGBA
            r, g, b = [int(x*255) for x in mcolors.to_rgb(color)]
            rgba_color = (r, g, b, 64)
        
        # Získáme indexy pixelů, kde je maska nenulová
        try:
            for y in range(pil_img.height):
                for x in range(pil_img.width):
                    try:
                        if mask[y, x]:  # Pokud je pixel součástí masky
                            draw.point((x, y), fill=rgba_color)
                    except IndexError:
                        continue  # Přeskočíme, pokud index je mimo rozsah
        except:
            st.warning(f"Maska {i} má jiný formát než očekáváno. Zobrazuji jako ohraničení.")
            # Pokud má maska jiný formát, zobrazíme jednoduchý box
            # Například pokud maska obsahuje souřadnice ohraničujícího rámečku [x, y, width, height]
            try:
                if len(mask) == 4:  # Formát [x, y, width, height]
                    x, y, w, h = mask
                    draw.rectangle([x, y, x+w, y+h], outline=rgba_color[:3], width=2)
            except:
                pass
    
    return result_img, object_colors

uploaded_file = st.file_uploader("Nahrajte obrázek", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Nahraný obrázek", use_container_width=True)
    
    if st.button("Segmentovat"):
        with st.spinner("Probíhá segmentace obrázku..."):
            img = image_utils.process_image(uploaded_file)
            masks, labels = segmentation.segment_image(img, config.HF_API_TOKEN)
            
            st.success(f"Segmentace dokončena! Nalezeno {len(masks)} objektů.")
            
            if len(masks) > 0:
                # Vizualizace segmentace
                segmented_img, object_colors = visualize_segmentation(img, masks, labels)
                
                # Zobrazení segmentovaného obrázku
                st.subheader("Segmentovaný obrázek")
                st.image(segmented_img, use_container_width=True)
                
                # Zobrazení seznamu nalezených objektů
                st.subheader("Nalezené objekty:")
                
                # Vytvoření sloupců pro přehlednější zobrazení
                num_cols = 3
                cols = st.columns(num_cols)
                
                # Zobrazení seznamu objektů
                for i, (label, color) in enumerate(zip(labels, object_colors)):
                    col_index = i % num_cols
                    with cols[col_index]:
                        st.markdown(
                            f"<div style='display: flex; align-items: center; margin-bottom: 5px;'>"
                            f"<div style='background-color: {color}; width: 15px; height: 15px; margin-right: 8px; border: 1px solid black;'></div>"
                            f"<span><b>{i+1}.</b> {label}</span>"
                            f"</div>", 
                            unsafe_allow_html=True
                        )
                
                # Možnost stáhnout segmentovaný obrázek
                buf = io.BytesIO()
                segmented_img.save(buf, format="PNG")
                st.download_button(
                    label="Stáhnout segmentovaný obrázek",
                    data=buf.getvalue(),
                    file_name="segmentovany_obrazek.png",
                    mime="image/png"
                )
            else:
                st.warning("Nebyly nalezeny žádné objekty k segmentaci.")