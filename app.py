import streamlit as st
from utils import image_utils
from models import llm, segmentation
from utils.prompt_utils import PromptBuilder
from utils.coco_class_map import COCO_CLASS_TRANSLATION, preprocess_class_name
import time
import config
import numpy as np
from PIL import Image

# Inicializace proměnných v session state
if "labels" not in st.session_state:
    st.session_state.labels = []
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "segmented_image" not in st.session_state:
    st.session_state.segmented_image = None
if "show_original" not in st.session_state:
    st.session_state.show_original = True
if "segment_attempt" not in st.session_state:
    st.session_state.segment_attempt = False
if "show_segment_button" not in st.session_state:
    st.session_state.show_segment_button = True


# Sidebar
st.sidebar.title("SegmenStory \U0001F9A7")
st.sidebar.markdown("**Strojové vidění \U0001F441 :**<br>Segmentace obrázků Mask2Former", unsafe_allow_html=True)
st.sidebar.markdown("**Evoluční antropologie \U0001F9B4 :**<br>Prof. Dr. Paleo Bagrstein, Ph.D.<br>\
                    ------------------------------------------------------", unsafe_allow_html=True)

# API klíče
st.sidebar.subheader("API Klíče")
st.sidebar.markdown("Pro generování textu si zvol mezi Perplexity a OpenAI.")
hf_api_key = st.sidebar.text_input("HuggingFace API klíč", value="", type="password")
per_api_key = st.sidebar.text_input("Perplexity API klíč", value="", type="password")
openai_api_key = st.sidebar.text_input("OpenAI API klíč", value="", type="password")

# Použití klíčů
api_key_hf = hf_api_key or config.HF_API_TOKEN
api_key_per = per_api_key or config.PER_API_TOKEN
api_key_openai = openai_api_key or config.OPENAI_API_KEY

# Výběr modelu pro generování textu
model_choice = st.sidebar.radio(
    "Vyber model pro generování příběhu:",
    ("Perplexity", "OpenAI"),
    index=0,
    key="llm_choice"
    )


# Hlavní část aplikace
uploaded_file = st.sidebar.file_uploader("Nahraj obrázek. Neboj zůstane jen u tebe! \U0000267B", type=["jpg", "jpeg", "png"])

if uploaded_file:
    if st.session_state.show_original:
        st.image(uploaded_file, caption="*No krásný to máš. S tím ti určitě pomáhala maminka, že?* \U0001F602", width=600)

    if uploaded_file and st.session_state.show_segment_button:
        if st.button("Segmentovat"):             
            with st.spinner("Už asi něco vidím, chvíli strpení... \U0001F441"):
                img = image_utils.process_image(uploaded_file)
            
                # Získání segmentovaného obrázku a tříd
                segmented_img, labels = segmentation.segment_image(img, config.HF_API_TOKEN)
                
                # Ujistit se, že segmented_img je ve správném formátu pro zobrazení
                if isinstance(segmented_img, np.ndarray):
                    segmented_img = Image.fromarray(segmented_img)
            
                # Uložení do session state
                st.session_state.segmented_image = segmented_img
                st.session_state.labels = labels
                st.session_state.show_original = False
                st.session_state.segment_attempt = True
                st.session_state.show_segment_button = False
                st.rerun()

# Zobrazení výsledků
if "segmented_image" in st.session_state and st.session_state.segmented_image is not None:
    # Převedení na PIL Image 
    if isinstance(st.session_state.segmented_image, np.ndarray):
        display_img = Image.fromarray(st.session_state.segmented_image)
    else:
        display_img = st.session_state.segmented_image
    
    # Zobrazení segmentovaného obrázku
    st.image(display_img, caption="Tak to vidí Mask2Former", width=600)

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
    
    st.write("Tady je seznam toho, co **Mask2Former** na obrázku rozpoznal. Vyber si jednu z položek v nabídce níže a profesor **Bagrstein** \
                 ti o ní poví její evoluční příběh.")      
    selected_class = st.selectbox(
    "Vyber třídu objektu:",
    unique_translated,
    index=0,
    key="main_label_selectbox",
    label_visibility="collapsed"
    ) 
    st.markdown("<br>Jen tak mezi námi:<br>**Mask2Former** neumí klasifikovat úplně všechno, co vidí, tak mu to odpusťme. Profesor **Bagrstein** \
                je ovšem odborník, takže zvládne okomentovat i segmenty, které nejsou zrovna středobodem lidské civilizace (třeba obloha, tráva, ...). \
                Nejvíc se ale vyžívá ve výkladu o různých předmětech a artefaktech \
                (jako je vidlička, pohovka, apod.), tak směle do toho!<br><br>", unsafe_allow_html=True)

    # Tlačítko pro generování
    if st.button("Zavolej profesora"):
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
            if model_choice == "Perplexity":
                llm_model = llm.PerplexityLLM()
            else:
                llm_model = llm.OpenAILLM()
            generated_text = llm_model.generate(prompt)

                
            # Zobrazení výsledku
            st.write(generated_text)
                
        except Exception as e:
            st.error(f"Chyba při generování: {str(e)}")
        
        my_bar.empty()

        st.markdown("<br><br>---------------------------------------------------------------------------------------------------------------------<br>\
                    *Admin: Jestli máš pocit, že tě profesor moc nepobavil nebo znalostně neobohatil, \
                    zavolej jej znovu. Můžeš si zvolit i jinou položku z nabídky.*", unsafe_allow_html=True)

# Hláška při neúspěšné segmentaci
if st.session_state.segment_attempt:
    st.write("Pokud nic nevidíš, nebo obrázek nemá zakreslené nalezené segmenty, klikni na Reset a pak znova na Segmentovat. Však to znáš z Windows. \U0001F926")

# Tlačítko pro resetování
if st.sidebar.button("Reset"):
    st.session_state.labels = []
    st.session_state.processed_image = None
    st.session_state.show_original = True
    st.session_state.segment_attempt = False
    st.session_state.segmented_image = None
    st.session_state.show_segment_button = True
    st.rerun()


