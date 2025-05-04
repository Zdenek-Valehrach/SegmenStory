from PIL import Image
from io import BytesIO
import numpy as np

def process_image(uploaded_file):
    """
    Zpracuje nahraný soubor obrázku
    
    Args:
        uploaded_file: Souborový objekt ze Streamlit file_uploader
        
    Returns:
        numpy array reprezentace obrazu
    """
    img_bytes = uploaded_file.getvalue()
    img = Image.open(BytesIO(img_bytes))
    
    # Zmenšení obrázku na mnohem menší velikost pro API (namísto 1920x1080) Hugging Face API má omezení velikosti payloadu
    img.thumbnail((512, 512))
    
    # Převod na RGB formát pokud obsahuje alfa kanál (průhlednost)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    return np.array(img)