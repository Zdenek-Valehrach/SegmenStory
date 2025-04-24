import streamlit as st
from utils import image_utils, prompt_utils
from models import segmentation, llm
import config
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import random
import io
import os

# Inicializace proměnných v session state
if "masks" not in st.session_state:
    st.session_state.masks = None
if "labels" not in st.session_state:
    st.session_state.labels = None
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "selected_segments" not in st.session_state:
    st.session_state.selected_segments = []
if "hidden_segments" not in st.session_state:
    st.session_state.hidden_segments = []
if "iteration_count" not in st.session_state:
    st.session_state.iteration_count = 0

st.title("SegmenStory")
st.write("Aplikace pro segmentaci obrázků pomocí pokročilých AI modelů")

# Funkce pro vizualizaci segmentovaných oblastí
def visualize_segmentation(image, masks, labels, selected_segments=None, hidden_segments=None):
    # Konverze numpy array na PIL Image
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image)
    else:
        pil_img = image
    
    # Vytvoření kopie pro kreslení
    result_img = pil_img.copy()
    draw = ImageDraw.Draw(result_img, 'RGBA')
    
    # Pokus o načtení fontu - použijeme výchozí font (pokud není dostupný, použije se výchozí)
    try:
        # Zkusíme použít základní font, který by měl být dostupný ve většině systémů
        font_size = 12
        font = ImageFont.truetype("Arial", font_size)
    except IOError:
        try:
            # Záložní možnost - použití defaultního fontu PIL
            font = ImageFont.load_default()
        except:
            font = None  # Když ani jeden není dostupný
    
    # Generování různých barev pro segmenty
    colors = list(mcolors.TABLEAU_COLORS.values())
    if len(masks) > len(colors):
        # Pokud je více masek než barev, generujeme náhodné barvy
        additional_colors = ['#{:06x}'.format(random.randint(0, 0xFFFFFF)) for _ in range(len(masks) - len(colors))]
        colors.extend(additional_colors)
    
    # Seznam pro uložení barev použitých pro každý objekt (pro legendu)
    object_colors = []
    
    # Seznam pro uložení informací o segmentech pro interaktivní zobrazení
    segments_info = []
    
    # Vykreslení segmentů
    for i, mask in enumerate(masks):
        # Skip hidden segments if specified
        if hidden_segments is not None and i in hidden_segments:
            continue
            
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
        
        # Převedeme na RGBA pro průhlednost a na RGB pro okraje
        if isinstance(color, str):
            # Převod hex barvy na RGBA
            if color.startswith('#'):
                color = color[1:]
            r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
            rgba_color = (r, g, b, 64)  # 64 je hodnota alpha (průhlednosti)
            rgb_color = (r, g, b)  # Plná barva pro ohraničení
        else:
            # Převod matplotlib barvy na RGBA
            r, g, b = [int(x*255) for x in mcolors.to_rgb(color)]
            rgba_color = (r, g, b, 64)
            rgb_color = (r, g, b)
        
        # Získáme label pro tento objekt
        label = labels[i] if i < len(labels) else f"Objekt {i+1}"
        
        # Určení bounding boxu pro masku (ohraničující obdélník)
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
            
            # Přidáme informaci o segmentu do seznamu pro interaktivní zobrazení
            segments_info.append({
                "id": i,
                "label": label,
                "coords": [x_min, y_min, width, height],
                "color": hex_color
            })
            
            # Vykreslení průhledné výplně masky
            if isinstance(mask, list) and len(mask) == 4:
                # Pro bounding box vykreslíme průhledný obdélník
                draw.rectangle([x_min, y_min, x_max, y_max], fill=(rgba_color))
            else:
                # Pro masku v podobě pole aplikujeme průhledné pixely
                for y in range(pil_img.height):
                    for x in range(pil_img.width):
                        try:
                            if mask[y, x]:  # Pokud je pixel součástí masky
                                draw.point((x, y), fill=rgba_color)
                        except IndexError:
                            continue  # Přeskočíme, pokud index je mimo rozsah
            
            # Check if this segment is selected
            is_selected = selected_segments is not None and i in selected_segments
            
            # Vykreslení ohraničujícího obdélníku s širší čárou - pro vybrané použijeme přerušovanou čáru
            outline_width = 3 if is_selected else 2
            for offset in range(outline_width):
                draw.rectangle(
                    [x_min-offset, y_min-offset, x_max+offset, y_max+offset], 
                    outline=rgb_color,
                    width=1
                )
                
                # Pro vybrané objekty vykreslíme přerušovanou čáru
                if is_selected:
                    # Vykreslíme dashed outline
                    dash_length = 5
                    for d in range(0, int((width + height) * 2), dash_length * 2):
                        # Top edge
                        if d < width:
                            draw.line(
                                [(x_min + d, y_min - offset), 
                                 (min(x_min + d + dash_length, x_max), y_min - offset)],
                                fill=(255, 255, 255),
                                width=1
                            )
                        # Right edge
                        elif d < width + height:
                            pos = d - width
                            draw.line(
                                [(x_max + offset, y_min + pos), 
                                 (x_max + offset, min(y_min + pos + dash_length, y_max))],
                                fill=(255, 255, 255),
                                width=1
                            )
                        # Bottom edge
                        elif d < width * 2 + height:
                            pos = d - (width + height)
                            draw.line(
                                [(x_max - pos, y_max + offset), 
                                 (max(x_max - pos - dash_length, x_min), y_max + offset)],
                                fill=(255, 255, 255),
                                width=1
                            )
                        # Left edge
                        else:
                            pos = d - (width * 2 + height)
                            draw.line(
                                [(x_min - offset, y_max - pos), 
                                 (x_min - offset, max(y_max - pos - dash_length, y_min))],
                                fill=(255, 255, 255),
                                width=1
                            )
            
            # Vykreslení textu s popiskem objektu
            # Přidáme tmavé pozadí pod text pro lepší čitelnost
            text_position = (x_min + 5, y_min + 5)
            
            # Zjistíme velikost textu pro pozadí
            if font:
                text_width, text_height = draw.textsize(label, font=font) if hasattr(draw, 'textsize') else (len(label) * 6, 12)
            else:
                text_width, text_height = len(label) * 6, 12  # Přibližný odhad
            
            # Vykreslení tmavého pozadí pod textem pro lepší čitelnost
            padding = 2
            background_box = [
                text_position[0] - padding, 
                text_position[1] - padding,
                text_position[0] + text_width + padding,
                text_position[1] + text_height + padding
            ]
            draw.rectangle(background_box, fill=(0, 0, 0, 180))  # Černé poloprůhledné pozadí
            
            # Vykreslení textu
            if font:
                draw.text(text_position, label, fill=(255, 255, 255), font=font)  # Bílý text
            else:
                draw.text(text_position, label, fill=(255, 255, 255))  # Bílý text bez specifického fontu
                
        except Exception as e:
            st.warning(f"Chyba při zpracování masky {i}: {str(e)}")
    
    return result_img, object_colors, segments_info

# Funkce pro zobrazení seznamu segmentů pro výběr
def list_select_segments(image, segments_info=None, key=None):
    """
    Zobrazí seznam segmentů pro výběr pomocí rolovací nabídky
    
    Args:
        image: PIL Image nebo numpy array s obrázkem
        segments_info: Seznam slovníků s informacemi o segmentech 
                      [{"id": 1, "label": "osoba", "coords": [x, y, w, h], "visible": True}, ...]
        key: Unikátní klíč pro Streamlit komponentu
    
    Returns:
        selected_segment: ID vybraného segmentu (pouze jeden)
        hidden_segments: Seznam ID skrytých segmentů
    """
    # Převod na PIL Image, pokud je potřeba
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Výchozí hodnoty pro segmenty
    if segments_info is None:
        segments_info = []
    
    # Převedeme NumPy datové typy na standardní Python typy pro JSON serializaci
    def numpy_to_python(obj):
        """Konvertuje NumPy datové typy na standardní Python typy pro JSON serializaci"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [numpy_to_python(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: numpy_to_python(v) for k, v in obj.items()}
        else:
            return obj
    
    segments_info = numpy_to_python(segments_info)
    
    # Vytvoříme unique klíč pro tuto instanci
    if key is None:
        key = "list_select_" + str(hash(image))
    
    # Inicializace stavů - nyní pouze jeden vybraný segment (ne seznam)
    if f"{key}_selected" not in st.session_state:
        st.session_state[f"{key}_selected"] = None
    
    if f"{key}_hidden" not in st.session_state:
        st.session_state[f"{key}_hidden"] = []
    
    # Filtrujeme pouze viditelné segmenty
    visible_segments = [seg for seg in segments_info if seg['id'] not in st.session_state[f"{key}_hidden"]]
    
    # Vytvoříme seznam možností pro select box - pouze název segmentu bez ID
    options = ["Žádný výběr"] + [seg['label'].split(' (ID:')[0] for seg in visible_segments]
    segment_ids = [None] + [seg['id'] for seg in visible_segments]
    
    # Najdeme index aktuálně vybraného segmentu v options
    selected_index = 0
    if st.session_state[f"{key}_selected"] is not None:
        try:
            selected_index = segment_ids.index(st.session_state[f"{key}_selected"])
        except ValueError:
            selected_index = 0
    
    st.write("### Seznam nalezených segmentů:")
    
    # Zobrazíme rolovací nabídku pro výběr segmentu
    selected_option = st.selectbox(
        "Vybrat segment:", 
        options,
        index=selected_index,
        key=f"select_{key}"
    )
    
    # Aktualizace vybraného segmentu podle výběru z rolovací nabídky
    selected_index = options.index(selected_option)
    st.session_state[f"{key}_selected"] = segment_ids[selected_index]
    
    # Tlačítka pro akce se segmenty
    col1, col2 = st.columns(2)
    
    with col1:
        # Tlačítko pro skrytí vybraného segmentu
        if st.button("Skrýt vybraný", key=f"hide_button_{key}", disabled=st.session_state[f"{key}_selected"] is None):
            # Přidáme vybraný segment mezi skryté
            if st.session_state[f"{key}_selected"] is not None:
                st.session_state[f"{key}_hidden"].append(st.session_state[f"{key}_selected"])
                st.session_state[f"{key}_selected"] = None
                st.rerun()
    
    with col2:
        # Tlačítko pro zobrazení všech segmentů
        if st.button("Zobrazit vše", key=f"show_all_button_{key}", disabled=not st.session_state[f"{key}_hidden"]):
            st.session_state[f"{key}_hidden"] = []
            st.rerun()
    
    # Zobrazíme informaci o skrytých segmentech
    if st.session_state[f"{key}_hidden"]:
        st.write(f"Skryté segmenty: {len(st.session_state[f'{key}_hidden'])}")
    
    # Vracíme aktuální stav segmentů - nyní pouze jeden vybraný segment (ne seznam)
    return [st.session_state[f"{key}_selected"]] if st.session_state[f"{key}_selected"] is not None else [], st.session_state[f"{key}_hidden"]

# Funkce pro zpracování kliknutí na segment
def toggle_segment(segment_id):
    if segment_id in st.session_state.selected_segments:
        st.session_state.selected_segments.remove(segment_id)
    else:
        st.session_state.selected_segments.append(segment_id)
    st.rerun()

# Funkce pro akce tlačítek
def hide_selected_segments():
    for segment_id in st.session_state.selected_segments:
        if segment_id not in st.session_state.hidden_segments:
            st.session_state.hidden_segments.append(segment_id)
    st.session_state.selected_segments = []
    st.rerun()

def show_all_segments():
    st.session_state.hidden_segments = []
    st.rerun()

def clear_selection():
    st.session_state.selected_segments = []
    st.rerun()

uploaded_file = st.file_uploader("Nahrajte obrázek", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Zobrazíme nahraný obrázek
    st.image(uploaded_file, caption="Nahraný obrázek", use_container_width=True)
    
    if st.button("Segmentovat"):
        with st.spinner("Probíhá segmentace obrázku..."):
            img = image_utils.process_image(uploaded_file)
            masks, labels = segmentation.segment_image(img, config.HF_API_TOKEN)
            
            # Uložíme výsledky do session state pro další použití
            st.session_state.masks = masks
            st.session_state.labels = labels
            st.session_state.processed_image = img
            st.session_state.selected_segments = []
            st.session_state.hidden_segments = []
            
            st.success(f"Segmentace dokončena! Nalezeno {len(masks)} objektů.")
            st.rerun()
    
    # Zpracování segmentovaného obrázku, pokud existuje
    if st.session_state.masks and len(st.session_state.masks) > 0:
        masks = st.session_state.masks
        labels = st.session_state.labels
        img = st.session_state.processed_image
        selected_segments = st.session_state.selected_segments
        hidden_segments = st.session_state.hidden_segments
        
        st.success(f"Segmentace dokončena! Nalezeno {len(masks)} objektů.")
        
        # Vytvoříme řádek tlačítek pro manipulaci se segmenty - jasně viditelné v UI
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Zobrazit všechny objekty", key="btn_show_all", use_container_width=True):
                show_all_segments()
        with col2:
            if st.button("Skrýt vybrané objekty", key="btn_hide_selected", use_container_width=True):
                hide_selected_segments()
        with col3:
            if st.button("Zrušit výběr", key="btn_clear_selection", use_container_width=True):
                clear_selection()
        
        # Informace o vybraných a skrytých objektech
        if selected_segments:
            selected_names = [labels[i] if i < len(labels) else f"Objekt {i+1}" for i in selected_segments]
            st.success(f"Vybrané objekty: {', '.join(selected_names)}")
        
        if hidden_segments:
            hidden_names = [labels[i] if i < len(labels) else f"Objekt {i+1}" for i in hidden_segments]
            st.info(f"Skryté objekty: {', '.join(hidden_names)}")
        
        # Vizualizace segmentace
        segmented_img, object_colors, segments_info = visualize_segmentation(
            img, masks, labels, 
            selected_segments=selected_segments, 
            hidden_segments=hidden_segments
        )
        
        # Zobrazení pouze jednoho obrázku s názvem "Výběr objektů"
        st.subheader("Výběr objektů")
        st.image(segmented_img, use_container_width=True)
        
        # Přidat tlačítko "Continue to iterate?" pro pokračování v iteraci segmentace
        if st.button("Continue to iterate?", key="btn_continue_iterate"):
            with st.spinner("Probíhá další iterace segmentace..."):
                # Zvýšíme počet iterací
                st.session_state.iteration_count += 1
                
                # Znovu spustíme segmentaci s aktuálním stavem (může přinést vylepšené výsledky)
                try:
                    # Předáme informaci o skrytých segmentech jako feedback pro model
                    refined_masks, refined_labels = segmentation.segment_image(
                        img, 
                        config.HF_API_TOKEN,
                        iteration=st.session_state.iteration_count,
                        hidden_segments=hidden_segments,
                        previous_labels=labels
                    )
                    
                    # Aktualizujeme výsledky v session state
                    st.session_state.masks = refined_masks
                    st.session_state.labels = refined_labels
                    
                    st.success(f"Iterace {st.session_state.iteration_count} dokončena! Nalezeno {len(refined_masks)} objektů.")
                except Exception as e:
                    st.error(f"Chyba při iteraci segmentace: {str(e)}")
                
                st.rerun()
        
        if st.session_state.iteration_count > 0:
            st.info(f"Počet provedených iterací: {st.session_state.iteration_count}")
        
        # Pouze jedna sekce pro výběr segmentů
        try:
            list_selected, list_hidden = list_select_segments(
                segmented_img,
                segments_info=segments_info,
                key="list_segments"
            )
            
            # Synchronizace stavu z komponenty výběru ze seznamu
            if list_selected != st.session_state.selected_segments:
                st.session_state.selected_segments = list_selected
                st.rerun()
            
            if list_hidden != st.session_state.hidden_segments:
                st.session_state.hidden_segments = list_hidden
                st.rerun()
                
        except Exception as e:
            st.error(f"Chyba při zpracování výběru ze seznamu: {str(e)}")
        
        # Možnost stáhnout segmentovaný obrázek
        buf = io.BytesIO()
        
        # Pro stažení vytvoříme obrázek bez skrytých segmentů
        visible_masks = [mask for i, mask in enumerate(masks) if i not in hidden_segments]
        visible_labels = [labels[i] if i < len(labels) else f"Objekt {i+1}" for i, _ in enumerate(visible_masks)]
        
        final_img, _, _ = visualize_segmentation(img, visible_masks, visible_labels)
        final_img.save(buf, format="PNG")
        
        st.download_button(
            label="Stáhnout segmentovaný obrázek",
            data=buf.getvalue(),
            file_name="segmentovany_obrazek.png",
            mime="image/png"
        )