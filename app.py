import streamlit as st
from utils import image_utils
from models import llm, segmentation
from utils.prompt_utils import PromptBuilder
from utils.coco_class_map import COCO_CLASS_TRANSLATION, preprocess_class_name
import time
import config
import numpy as np

# Inicializace proměnných v session state
if "labels" not in st.session_state:
    st.session_state.labels = []
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "show_original" not in st.session_state:
    st.session_state.show_original = True
if "segment_attempt" not in st.session_state:
    st.session_state.segment_attempt = False


# Sidebar
st.sidebar.title("SegmenStory \U0001F9A7")
st.sidebar.markdown("**Strojové vidění \U0001F441 :**<br>Segmentace obrázků Mask2Former", unsafe_allow_html=True)
st.sidebar.markdown("**Evoluční antropologie \U0001F9B4 :**<br>Prof. Dr. Paleo Bagrstein, Ph.D.", unsafe_allow_html=True)

# Hlavní část aplikace
uploaded_file = st.sidebar.file_uploader("Nahraj obrázek. Neboj nikomu ho neukáži \U0000267B", type=["jpg", "jpeg", "png"])

if uploaded_file:
    if st.session_state.show_original:
        st.image(uploaded_file, caption="*No krásný to máš. S tím ti určitě pomáhala maminka, že?* \U0001F602", use_container_width=True)

    if st.sidebar.button("Segmentovat \U0001F441"):
        with st.spinner("Už něco vidím... \U0001F441"):
            img = image_utils.process_image(uploaded_file)
            _, labels = segmentation.segment_image(img, config.HF_API_TOKEN)
            st.session_state.labels = list(set(labels))  # Deduplikace tříd
            st.session_state.processed_image = img
            st.session_state.show_original = False
            st.session_state.segment_attempt = True
            st.rerun()

# Zobrazení výsledků
if st.session_state.labels:
    st.session_state.segment_attempt = False
    unique_labels = sorted(st.session_state.labels)
    # Překlad a deduplikace
    translated_labels = []
    for label in unique_labels:
        # Zpracování speciálních přípon
        translated = preprocess_class_name(label)
        
        # Ošetření neznámých tříd
        if translated == label:
            translated += " (nepodařilo se identifikovat)"
            
        translated_labels.append(translated)
    
    # Odstranění duplicit po překladu
    unique_translated = list(set(translated_labels))
    
    st.subheader("Dalo to zabrat, ale něco vidím \U0001F441")        # \U0001F4A1  
    st.write("Tady je seznam toho co Mask2Famer našel. Vyber si jednu a profesor Bagrstein ti k ní něco poví.")
    selected_class = st.selectbox(
        "",
        unique_translated,
        index=0,
        key="main_label_selectbox"
    )

    # Tlačítko pro generování
    if st.button("Zavolej profesora  \U0001F9B4"):
        progress_text = "Už to kopu, vydrž... \U0001F69C"
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)

        try:
            # Vytvoření promptu
            builder = PromptBuilder()
            prompt = builder.build(selected_class)
                
            # Volání LLM
            llm = llm.PerplexityLLM()
            generated_text = llm.generate(prompt)
                
            # Zobrazení výsledku
            # st.subheader(f"Pravda o: {selected_class}")
            st.write(generated_text)
                
        except Exception as e:
            st.error(f"Chyba při generování: {str(e)}")
        
        my_bar.empty()

# Hláška po segmentaci
if st.session_state.segment_attempt:
    st.write("Pokud nic nevidíš, klikni na Reset a pak znova na Segmentovat. Však to znáš z Windows. \U0001F926")

# Tlačítko pro resetování
if st.sidebar.button("Reset"):
    st.session_state.labels = []
    st.session_state.processed_image = None
    st.session_state.show_original = True
    st.session_state.segment_attempt = False
    st.rerun()


