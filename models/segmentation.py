import requests
import streamlit as st
import base64
from io import BytesIO
from PIL import Image

SEGMENTATION_MODEL = {
    "id": "facebook/mask2former-swin-large-coco-panoptic",
    "description": "Mask2Former - pokročilý model pro segmentaci objektů"
}

def segment_image(image_array, hf_token):
    """
    Segmentuje obrázek pomocí Mask2Former modelu přes Hugging Face API
    Vrací pouze unikátní třídy objektů
    """
    # Příprava obrázku pro API
    pil_img = Image.fromarray(image_array)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=90)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # API požadavek
    url = f"https://api-inference.huggingface.co/models/{SEGMENTATION_MODEL['id']}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    with st.spinner(f"{SEGMENTATION_MODEL['description']} na tom mrká, jak ďas..."):
        try:
            response = requests.post(url, headers=headers, json={"inputs": img_str}, timeout=30)
            
            if response.status_code == 200:
                results = response.json()
                labels = []
                
                # Zpracování výsledků
                if isinstance(results, list):
                    for segment in results:
                        if "label" in segment:
                            label = segment["label"].split(":")[-1].strip()
                            labels.append(label)
                
                return [], list(set(labels))  # Vracíme pouze unikátní třídy
            
            st.error(f"Chyba API: {response.status_code}")
            return [], []
        
        except Exception as e:
            st.error(f"Chyba při komunikaci s API: {str(e)}")
            return [], []
