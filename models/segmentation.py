import requests
import numpy as np
import streamlit as st
import base64
import time
from io import BytesIO
from PIL import Image

# Definujeme několik alternativních modelů pro případ, že primární model není dostupný
SEGMENTATION_MODELS = [
    {
        "id": "facebook/mask2former-swin-large-coco-panoptic",
        "task": "image-segmentation",
        "description": "Mask2Former - pokročilý model pro přesnou segmentaci objektů v obraze"
    },
    {
        "id": "facebook/detr-resnet-50-panoptic",  # Záložní model
        "task": "image-segmentation",
        "description": "DETR model pro segmentaci obrazu"
    }
]

def segment_image(image_array, hf_token, max_retries=2, retry_delay=2):
    """
    Segments an image using a segmentation model via Hugging Face API
    with retry logic and fallback models
    
    Args:
        image_array: numpy array representation of the image
        hf_token: Hugging Face API token
        max_retries: Maximum number of retry attempts per model
        retry_delay: Delay between retries in seconds
        
    Returns:
        masks: list of segmentation masks
        labels: list of corresponding labels
    """
    # Převod numpy array zpět na PIL Image a poté na base64
    pil_img = Image.fromarray(image_array)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=85)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # Zjednodušený payload pro modely
    payload = {"inputs": img_str}
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Zkusíme postupně všechny definované modely
    for model_idx, model_info in enumerate(SEGMENTATION_MODELS):
        model_name = model_info["id"].split("/")[-1]
        url = f"https://api-inference.huggingface.co/models/{model_info['id']}"
        
        # Pro každý model zkusíme několik pokusů
        for attempt in range(max_retries):
            try:
                with st.spinner(f"Kontaktuji Hugging Face API... (Model {model_idx+1}/{len(SEGMENTATION_MODELS)}, Pokus {attempt+1}/{max_retries})"):
                    st.info(f"Odesílám požadavek na segmentaci obrázku pomocí modelu {model_name}...")
                    
                    response = requests.post(url, headers=headers, json=payload, timeout=30)
                
                    # Vypisovat status_code a obsah odpovědi pro diagnostiku
                    st.write(f"API Status Code: {response.status_code}")
                    
                    # Úspěšná odpověď
                    if response.status_code == 200:
                        try:
                            results = response.json()
                            st.success(f"Úspěšně získána odpověď z API! (model {model_name})")
                            
                            # Zpracování výsledků
                            masks = []
                            labels = []
                            
                            # Zpracování výsledků z různých typů modelů
                            if isinstance(results, list):
                                for i, segment in enumerate(results):
                                    if "mask" in segment:
                                        masks.append(segment["mask"])
                                        # Použijeme label pokud existuje, jinak generický název
                                        if "label" in segment:
                                            label = segment["label"]
                                            # Odstraníme číselný prefix, pokud existuje (např. "29: potted plant" -> "potted plant")
                                            if ":" in label:
                                                label = label.split(":", 1)[1].strip()
                                            labels.append(label)
                                        else:
                                            labels.append(f"Objekt {i+1}")
                                    # Alternativní formát pro některé verze modelů
                                    elif "segmentation" in segment:
                                        masks.append(segment["segmentation"])
                                        label = segment.get("label", f"Objekt {i+1}")
                                        labels.append(label)
                            # Alternativní formát - jediný objekt s maskami
                            elif isinstance(results, dict):
                                if "masks" in results:
                                    for i, mask in enumerate(results["masks"]):
                                        masks.append(mask)
                                        # Pokud existují labely, použijeme je
                                        if "labels" in results and i < len(results["labels"]):
                                            labels.append(results["labels"][i])
                                        else:
                                            labels.append(f"Objekt {i+1}")
                                # Další možný formát
                                elif "segmentation" in results:
                                    masks.append(results["segmentation"])
                                    label = results.get("label", "Objekt 1")
                                    labels.append(label)
                            
                            if masks:
                                return masks, labels
                            else:
                                st.warning(f"Model {model_name} nevrátil žádné rozpoznatelné objekty. Zkusím jiný model.")
                                break
                        except Exception as e:
                            st.error(f"Nelze zpracovat odpověď jako JSON: {str(e)}")
                            if attempt < max_retries - 1:
                                st.warning(f"Zkouším znovu za {retry_delay} sekund...")
                                time.sleep(retry_delay)
                            else:
                                st.warning(f"Zkouším jiný model...")
                    # Kód 503 - služba nedostupná
                    elif response.status_code == 503:
                        st.warning("Hugging Face API je momentálně přetížené nebo nedostupné.")
                        if "html" in response.text.lower():
                            st.warning("API vrátilo HTML stránku místo očekávané odpovědi.")
                        if attempt < max_retries - 1:
                            st.info(f"Zkouším znovu za {retry_delay} sekund...")
                            time.sleep(retry_delay)
                        else:
                            st.info(f"Zkouším jiný model...")
                    # Jiná chyba
                    else:
                        st.error(f"API vrátilo chybu: {response.status_code}")
                        if attempt < max_retries - 1:
                            st.info(f"Zkouším znovu za {retry_delay} sekund...")
                            time.sleep(retry_delay)
                        else:
                            st.info(f"Zkouším jiný model...")
            
            except requests.exceptions.RequestException as e:
                st.error(f"Chyba při komunikaci s API: {str(e)}")
                if attempt < max_retries - 1:
                    st.info(f"Zkouším znovu za {retry_delay} sekund...")
                    time.sleep(retry_delay)
                else:
                    st.info(f"Zkouším jiný model...")
    
    # Pokud všechny modely selhaly
    st.error("Všechny modely selhaly. Hugging Face API může být momentálně nedostupné.")
    st.info("Zkuste to prosím později nebo zkontrolujte vaše připojení k internetu.")
    return [], []