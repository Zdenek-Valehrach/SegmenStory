import requests
import numpy as np
import streamlit as st
import base64
from io import BytesIO
from PIL import Image

def segment_image(image_array, hf_token):
    """
    Segments an image using Segment Anything Model (SAM) via Hugging Face API
    
    Args:
        image_array: numpy array representation of the image
        hf_token: Hugging Face API token
        
    Returns:
        masks: list of segmentation masks
        labels: list of corresponding labels
    """
    # Změněn model na Facebook Segment Anything, který podporuje image-segmentation
    url = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50-panoptic"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    try:
        # Převod numpy array zpět na PIL Image a poté na base64
        pil_img = Image.fromarray(image_array)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Přidání explicitního zadání typu úlohy - image-segmentation
        payload = {
            "inputs": img_str,
            "parameters": {
                "task": "image-segmentation"
            }
        }
        
        with st.spinner("Kontaktuji Hugging Face API..."):
            st.info("Odesílám požadavek na segmentaci obrázku...")
            response = requests.post(url, headers=headers, json=payload)
        
            # Vypisovat status_code a obsah odpovědi pro diagnostiku
            st.write(f"API Status Code: {response.status_code}")
            
            # Kontrola validní odpovědi
            if response.status_code != 200:
                st.error(f"API vrátilo chybu: {response.text}")
                return [], []
                
            # Pokus o parsování JSON
            try:
                results = response.json()
                st.success("Úspěšně získána odpověď z API!")
                
                # Zpracování výsledků v očekávaném formátu API
                # DETR model vrací segmenty s labely
                masks = []
                labels = []
                
                if isinstance(results, list):
                    for segment in results:
                        if "mask" in segment:
                            masks.append(segment["mask"])
                            # Použijeme ID nebo label pokud existuje
                            label = segment.get("label", f"Objekt {len(labels)+1}")
                            labels.append(label)
                else:
                    st.warning("Neočekávaný formát dat z API")
                    st.write(f"Odpověď API: {str(results)[:500]}...")
                
                if not masks:
                    st.warning("API nevrátilo žádné masky.")
                
                return masks, labels
                
            except Exception as e:
                st.error(f"Nelze zpracovat odpověď jako JSON: {str(e)}")
                st.write(f"Obsah odpovědi: {response.text[:500]}...")
                return [], []
            
    except Exception as e:
        st.error(f"Chyba při komunikaci s API: {str(e)}")
        import traceback
        st.write(f"Detaily chyby: {traceback.format_exc()}")
        return [], []