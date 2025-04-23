import requests
import numpy as np

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
    url = "https://api-inference.huggingface.co/models/facebook/sam-vit-b"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Convert numpy array to list for JSON serialization
    response = requests.post(url, headers=headers, json={"inputs": image_array.tolist()})
    data = response.json()
    
    masks = data.get("masks", [])
    labels = [f"Objekt {i+1}" for i in range(len(masks))]
    
    return masks, labels