import requests
import time
import streamlit as st
from config import HF_API_TOKEN  # Import tokenu z config.py

def generate_text(prompt, hf_token=HF_API_TOKEN):
    """
    Generates text using Hugging Face API with reliable fallback models
    """
    # Seznam ověřených modelů dostupných přes Inference API
    models = [
        "HuggingFaceH4/zephyr-7b-beta",        # Rychlý a kvalitní
        "CohereForAI/aya-23-35B",              # Výborná čeština
        "mistralai/Mistral-7B-Instruct-v0.2",  # Osvědčený model
        "Qwen/Qwen1.5-7B-Chat",                # Multilingvální
        "google/flan-t5-xxl"                   # Záložní jednoduchý model
    ]
    
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    
    # Optimalizovaný payload pro chatové modely
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.2
        }
    }

    for model in models:
        url = f"https://api-inference.huggingface.co/models/{model}"
        st.info(f"Attempting to generate using: {model}")
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=45  # Zvýšený timeout pro větší modely
            )
            
            # Zpracování odpovědi
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Unifikované zpracování různých formátů odpovědí
                    if isinstance(data, list):
                        return data[0].get('generated_text', data[0].get('text', ''))
                    if isinstance(data, dict):
                        return data.get('generated_text', data.get('text', ''))
                    return str(data)
                except Exception as e:
                    st.warning(f"Response parsing error: {str(e)}")
                    continue
            
            # Specifické chybové stavy
            elif response.status_code == 503:
                wait_time = response.json().get('estimated_time', 30)
                st.info(f"Model loading - retrying in {wait_time}s...")
                time.sleep(wait_time + 5)
                continue
                
            elif response.status_code in [401, 403]:
                st.error("Authorization failed - check your API token")
                break
                
            else:
                st.warning(f"HTTP {response.status_code} - trying next model")
                continue
                
        except requests.exceptions.RequestException as e:
            st.warning(f"Connection error: {str(e)}")
            time.sleep(2)
            continue

    st.error("All models failed. Check token and model availability.")
    return "Text generation failed. Please try again later."
