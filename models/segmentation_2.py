import requests
import numpy as np
import streamlit as st
import base64
from io import BytesIO
from PIL import Image

# Definice modelu SAM
SAM_MODEL = {
    "id": "facebook/sam-vit-huge",
    "description": "Segment Anything Model (SAM) - segmentuje všechny objekty bez ohledu na třídu"
}

# SAM_MODEL = {
#     "id": "facebook/sam-vit-base",  # Místo "huge"
#     "description": "Segment Anything Model (SAM) - base verze"
# }

def segment_image(image_array, hf_token):
    """
    Segmentuje obrázek pomocí modelu SAM přes Hugging Face API.
    Vrací masky a generické labely 'Objekt X'.
    """
    try:
        pil_img = Image.fromarray(image_array)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=90)
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        payload = {"inputs": img_str}
        headers = {"Authorization": f"Bearer {hf_token}"}

        url = f"https://api-inference.huggingface.co/models/{SAM_MODEL['id']}"
        
        with st.spinner(f"Segmentuji obrázek modelem {SAM_MODEL['description']}..."):
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            # Nový debug výpis
            st.write(f"API Status: {response.status_code}")
            st.write(f"API Response: {response.text}")
            
            if response.status_code == 200:
                results = response.json()
                masks, labels = [], []
                
                if isinstance(results, list):
                    for segment in results:
                        mask = segment.get("mask")
                        if mask:
                            masks.append(mask)
                            labels.append(f"Objekt {len(masks)}")
                
                if masks:
                    st.success("Segmentace úspěšná pomocí SAM")
                    return masks, labels
                else:
                    st.error("SAM nenašel žádné segmenty")
                    return [], []
            else:
                # Zpracování chybové zprávy
                error_msg = response.json().get('error', 'Unknown error')
                st.error(f"Detail chyby: {error_msg}")
                return [], []

    except requests.exceptions.RequestException as e:
        st.error(f"Chyba připojení k API: {str(e)}")
    except Exception as e:
        st.error(f"Neočekávaná chyba: {str(e)}")
    
    return [], []
