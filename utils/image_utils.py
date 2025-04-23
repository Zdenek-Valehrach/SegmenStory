from PIL import Image
from io import BytesIO
import numpy as np

def process_image(uploaded_file):
    """
    Process an uploaded image file
    
    Args:
        uploaded_file: File object from Streamlit file_uploader
        
    Returns:
        numpy array representation of the image
    """
    img_bytes = uploaded_file.getvalue()
    img = Image.open(BytesIO(img_bytes))
    
    # Zmenšení obrázku na mnohem menší velikost pro API (namísto 1920x1080)
    # Hugging Face API má omezení velikosti payloadu
    img.thumbnail((512, 512))
    
    # Převod na RGB formát pokud obsahuje alfa kanál (průhlednost)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    return np.array(img)