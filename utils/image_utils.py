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
    
    # Resize large images to reasonable dimensions
    img.thumbnail((1920, 1080))
    
    return np.array(img)