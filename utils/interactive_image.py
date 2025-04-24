import streamlit as st
import streamlit.components.v1 as components
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import json

def get_image_base64(image):
    """Převádí PIL image na base64 pro zobrazení v HTML"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Pomocná funkce pro konverzi NumPy typů na standardní Python typy
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

def interactive_image(image, segments_info=None, key=None):
    """
    Zobrazí interaktivní obrázek, který umožňuje klikáním vybírat segmenty
    
    Args:
        image: PIL Image nebo numpy array s obrázkem
        segments_info: Seznam slovníků s informacemi o segmentech 
                      [{"id": 1, "label": "osoba", "coords": [x, y, w, h], "visible": True}, ...]
        key: Unikátní klíč pro Streamlit komponentu
    
    Returns:
        selected_segments: Seznam ID vybraných segmentů
        hidden_segments: Seznam ID skrytých segmentů
    """
    # Převod na PIL Image, pokud je potřeba
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Výchozí hodnoty pro segmenty
    if segments_info is None:
        segments_info = []
    
    # Převedeme NumPy datové typy na standardní Python typy
    segments_info = numpy_to_python(segments_info)
    
    # Vytvoříme unique klíč pro tuto instanci
    if key is None:
        key = "interactive_image_" + str(hash(image))
    
    # Získáme aktuálně vybrané segmenty z globálního session_state
    selected_from_app = st.session_state.get('selected_segments', [])
    hidden_from_app = st.session_state.get('hidden_segments', [])
    
    # Převod na base64 pro HTML
    img_base64 = get_image_base64(image)
    
    # State pro ukládání vybraných a skrytých segmentů
    if f"{key}_selected" not in st.session_state:
        st.session_state[f"{key}_selected"] = selected_from_app.copy()
    
    if f"{key}_hidden" not in st.session_state:
        st.session_state[f"{key}_hidden"] = hidden_from_app.copy()
    
    # Zajistíme, že existuje klíč pro flag pro rerun
    if f"{key}_needs_update" not in st.session_state:
        st.session_state[f"{key}_needs_update"] = False
    
    # Vytvoříme komunikační kanál pro předávání událostí z JavaScript do Python
    if f"{key}_last_event" not in st.session_state:
        st.session_state[f"{key}_last_event"] = None
    
    # Aktualizujeme vybrané segmenty z hlavní aplikace (pokud se změnily)
    if selected_from_app != st.session_state[f"{key}_selected"]:
        st.session_state[f"{key}_selected"] = selected_from_app.copy()
    
    # Aktualizujeme skryté segmenty z hlavní aplikace
    if hidden_from_app != st.session_state[f"{key}_hidden"]:
        st.session_state[f"{key}_hidden"] = hidden_from_app.copy()
    
    # Zakódujeme informace o segmentech do JSON pro JavaScript
    segments_json = json.dumps(segments_info)
    
    # HTML a JavaScript pro interaktivní obrázek
    html = f"""
    <style>
        .interactive-image-container {{
            position: relative;
            max-width: 100%;
            margin: 0 auto;
        }}
        .segment-box {{
            position: absolute;
            cursor: pointer;
            border: 2px solid;
            box-sizing: border-box;
            transition: all 0.2s;
        }}
        .segment-box:hover {{
            filter: brightness(1.2);
        }}
        .segment-label {{
            position: absolute;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 2px 4px;
            font-size: 12px;
            pointer-events: none;
        }}
        .segment-selected {{
            border-width: 3px;
            border-style: dashed;
        }}
        .segment-hidden {{
            display: none;
        }}
        .controls {{
            margin-top: 10px;
            display: flex;
            justify-content: center;
            gap: 10px;
        }}
        .control-button {{
            padding: 5px 10px;
            background-color: #f0f0f0;
            border: 1px solid #ccc;
            border-radius: 4px;
            cursor: pointer;
        }}
        .control-button:hover {{
            background-color: #e0e0e0;
        }}
    </style>
    
    <div class="interactive-image-container" id="container-{key}">
        <img src="data:image/png;base64,{img_base64}" style="width: 100%; display: block;" />
        <div id="segments-{key}"></div>
    </div>
    
    <script>
        // Informace o segmentech
        const segmentsInfo = {segments_json};
        
        // Stav segmentů
        let selectedSegments = {json.dumps(st.session_state[f"{key}_selected"])};
        let hiddenSegments = {json.dumps(st.session_state[f"{key}_hidden"])};
        
        // Funkce pro aktualizaci stavu
        function updateState() {{
            const data = {{
                selected: selectedSegments,
                hidden: hiddenSegments
            }};
            
            // Předání dat zpět do Streamlit pomocí custom eventu
            // Tento approach používá window.parent pro komunikaci s rodičovským Streamlit
            const stringifiedData = JSON.stringify(data);
            window.parent.postMessage({{
                type: "streamlit:setComponentValue",
                value: stringifiedData
            }}, "*");
            
            // Ukládáme stav do sessionStorage pro zachování mezi refreshi
            sessionStorage.setItem('{key}_selected', JSON.stringify(selectedSegments));
            sessionStorage.setItem('{key}_hidden', JSON.stringify(hiddenSegments));
        }}
        
        // Obnovení stavu z sessionStorage
        const savedSelected = sessionStorage.getItem('{key}_selected');
        const savedHidden = sessionStorage.getItem('{key}_hidden');
        
        if (savedSelected) {{
            selectedSegments = JSON.parse(savedSelected);
        }}
        
        if (savedHidden) {{
            hiddenSegments = JSON.parse(savedHidden);
        }}
        
        // Vykreslení segmentů
        function renderSegments() {{
            const container = document.getElementById('container-{key}');
            const segmentsDiv = document.getElementById('segments-{key}');
            segmentsDiv.innerHTML = '';
            
            const img = container.querySelector('img');
            const containerRect = container.getBoundingClientRect();
            const imgRect = img.getBoundingClientRect();
            
            const scaleX = img.width / img.naturalWidth;
            const scaleY = img.height / img.naturalHeight;
            
            segmentsInfo.forEach((segment) => {{
                if (hiddenSegments.includes(segment.id)) {{
                    return; // Přeskočit skryté segmenty
                }}
                
                const [x, y, w, h] = segment.coords;
                
                const box = document.createElement('div');
                box.className = 'segment-box';
                if (selectedSegments.includes(segment.id)) {{
                    box.classList.add('segment-selected');
                }}
                
                box.style.left = `${{x * scaleX}}px`;
                box.style.top = `${{y * scaleY}}px`;
                box.style.width = `${{w * scaleX}}px`;
                box.style.height = `${{h * scaleY}}px`;
                box.style.borderColor = segment.color;
                
                box.setAttribute('data-id', segment.id);
                box.onclick = function(e) {{
                    e.stopPropagation();
                    toggleSegment(segment.id);
                }};
                
                // Přidání popisku
                const label = document.createElement('div');
                label.className = 'segment-label';
                label.textContent = segment.label;
                label.style.top = '5px';
                label.style.left = '5px';
                
                box.appendChild(label);
                segmentsDiv.appendChild(box);
            }});
        }}
        
        // Přepínání výběru segmentu
        function toggleSegment(id) {{
            const index = selectedSegments.indexOf(id);
            if (index === -1) {{
                selectedSegments.push(id);
            }} else {{
                selectedSegments.splice(index, 1);
            }}
            renderSegments();
            updateState();
        }}
        
        // Inicializace při načtení
        window.addEventListener('load', function() {{
            renderSegments();
            
            // Přidáme event listener pro změnu velikosti okna
            window.addEventListener('resize', function() {{
                renderSegments();
            }});
            
            // Explicitní aktualizace stavu při načtení
            updateState();
        }});
        
        // Naslouchání na zprávy ze Streamlit
        window.addEventListener('message', function(event) {{
            if (event.data.type === 'streamlit:render') {{
                renderSegments();
            }}
        }});
    </script>
    """
    
    # Používáme HTML komponentu s callback funkcí
    components_value = components.html(
        html, 
        height=image.height + 50,  # +50 pro kontrolní tlačítka
        scrolling=False
    )
    
    # Zpracování dat vrácených z JavaScript
    if components_value:
        try:
            data = json.loads(components_value)
            # Aktualizujeme stav v Streamlit
            st.session_state[f"{key}_selected"] = data.get("selected", [])
            st.session_state[f"{key}_hidden"] = data.get("hidden", [])
            
            # Důležitá změna: aktualizujeme i globální stav selected_segments
            if "selected" in data:
                st.session_state.selected_segments = data.get("selected", [])
            
            # Nastavíme flag pro rerun
            if st.session_state[f"{key}_last_event"] != components_value:
                st.session_state[f"{key}_last_event"] = components_value
                st.session_state[f"{key}_needs_update"] = True
        except Exception as e:
            # V případě chyby při parsování JSON
            pass
    
    # Vrátíme aktuální stav segmentů - důležité pro předání do hlavní aplikace
    selected = st.session_state[f"{key}_selected"]
    hidden = st.session_state[f"{key}_hidden"]
    
    # Pokud potřebujeme aktualizovat stránku kvůli JavaScript eventu
    if st.session_state[f"{key}_needs_update"]:
        st.session_state[f"{key}_needs_update"] = False
        # Zajistíme, že se session_state.selected_segments aktualizuje
        st.session_state.selected_segments = selected
    
    # Vracíme aktuální stav segmentů
    return selected, hidden

def list_select_segments(image, segments_info=None, key=None):
    """
    Zobrazí seznam segmentů pro výběr ze seznamu místo klikání na obrázek
    
    Args:
        image: PIL Image nebo numpy array s obrázkem
        segments_info: Seznam slovníků s informacemi o segmentech 
                      [{"id": 1, "label": "osoba", "coords": [x, y, w, h], "visible": True}, ...]
        key: Unikátní klíč pro Streamlit komponentu
    
    Returns:
        selected_segments: Seznam ID vybraných segmentů
        hidden_segments: Seznam ID skrytých segmentů
    """
    # Převod na PIL Image, pokud je potřeba
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Výchozí hodnoty pro segmenty
    if segments_info is None:
        segments_info = []
    
    # Převedeme NumPy datové typy na standardní Python typy
    segments_info = numpy_to_python(segments_info)
    
    # Vytvoříme unique klíč pro tuto instanci
    if key is None:
        key = "list_select_" + str(hash(image))
    
    # Inicializace stavů
    if f"{key}_selected" not in st.session_state:
        st.session_state[f"{key}_selected"] = []
    
    if f"{key}_hidden" not in st.session_state:
        st.session_state[f"{key}_hidden"] = []
    
    # Odstraněno zobrazení obrázku, aby se nezobrazoval dvakrát
    # (obrázek se již zobrazuje v hlavní aplikaci)
    
    # Filtrujeme pouze viditelné segmenty
    visible_segments = [seg for seg in segments_info if seg['id'] not in st.session_state[f"{key}_hidden"]]
    
    # Zobrazíme tlačítka pro každý segment (implementace pomocí buttonů místo multiselectu)
    st.write("### Seznam nalezených segmentů:")
    
    # Rozdělíme segmenty do řádků po 3 tlačítkách
    segment_buttons = []
    for i in range(0, len(visible_segments), 3):
        row = visible_segments[i:i+3]
        cols = st.columns(len(row))
        for idx, segment in enumerate(row):
            is_selected = segment['id'] in st.session_state[f"{key}_selected"]
            button_label = segment['label']  # Display only the label without ID
            button_color = "primary" if is_selected else "secondary"
            
            # V novějších verzích Streamlit můžeme použít use_container_width=True
            if cols[idx].button(
                button_label, 
                key=f"btn_{key}_{segment['id']}",
                type=button_color
            ):
                # Přepnutí stavu výběru
                if segment['id'] in st.session_state[f"{key}_selected"]:
                    st.session_state[f"{key}_selected"].remove(segment['id'])
                else:
                    st.session_state[f"{key}_selected"].append(segment['id'])
                st.rerun()  # Používáme st.rerun() místo st.experimental_rerun()
    
    # Tlačítka pro akce se segmenty
    col1, col2 = st.columns(2)
    
    with col1:
        # Tlačítko pro skrytí vybraných segmentů
        if st.button("Skrýt vybrané", key=f"hide_button_{key}", disabled=not st.session_state[f"{key}_selected"]):
            # Přidáme vybrané segmenty mezi skryté
            st.session_state[f"{key}_hidden"] = list(set(st.session_state[f"{key}_hidden"] + st.session_state[f"{key}_selected"]))
            # Vyčistíme výběr
            st.session_state[f"{key}_selected"] = []
            st.rerun()  # Používáme st.rerun() místo st.experimental_rerun()
    
    with col2:
        # Tlačítko pro zobrazení všech segmentů
        if st.button("Zobrazit vše", key=f"show_all_button_{key}", disabled=not st.session_state[f"{key}_hidden"]):
            st.session_state[f"{key}_hidden"] = []
            st.rerun()  # Používáme st.rerun() místo st.experimental_rerun()
    
    # Zobrazíme informaci o skrytých segmentech
    if st.session_state[f"{key}_hidden"]:
        st.write(f"Skryté segmenty: {len(st.session_state[f'{key}_hidden'])}")
    
    # Zobrazíme informaci o aktuálně vybraných segmentech
    if st.session_state[f"{key}_selected"]:
        st.write(f"Vybrané segmenty: {', '.join([str(id) for id in st.session_state[f'{key}_selected']])}")
    
    # Vracíme aktuální stav segmentů
    return st.session_state[f"{key}_selected"], st.session_state[f"{key}_hidden"]