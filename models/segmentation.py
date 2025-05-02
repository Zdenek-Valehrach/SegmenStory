import requests
import streamlit as st
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import colorsys

SEGMENTATION_MODEL = {
    "id": "facebook/mask2former-swin-large-coco-panoptic",
    "description": "Mask2Former - pokročilý model pro segmentaci objektů"
}

def draw_masks(image: Image.Image, boxes: list, color='red', width=3):
    """Vykreslení segmentační masky na obrázek"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for box in boxes:
        # box je [x1, y1, x2, y2]
        draw.rectangle(box, outline=color, width=width)
    return img

def decode_base64_mask(base64_string):
    """Dekódování Base64 kódovaný obrázek masky"""
    try:
        # Dekódování Base64 řetězce na bytes
        mask_bytes = base64.b64decode(base64_string)
        # Převod na obrázek
        mask_img = Image.open(BytesIO(mask_bytes))
        return mask_img
    except Exception as e:
        st.error(f"Chyba při dekódování masky: {str(e)}")
        return None

def apply_colored_mask(image, mask_img, color=(255, 0, 0, 128)):
    """Aplikujeme barevnou masku na obrázek"""
    # Ujistíme se, že obrázek je v režimu RGBA
    img_rgba = image.convert("RGBA")
    
    # Převedeme masku na správnou velikost, pokud není stejná
    if mask_img.size != image.size:
        mask_img = mask_img.resize(image.size)
    
    # Vytvoříme barevnou masku
    colored_mask = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_pixels = mask_img.convert("L")  # Převedeme na černobílou
    
    # Překreslíme masku danou barvou
    for y in range(image.height):
        for x in range(image.width):
            if mask_pixels.getpixel((x, y)) > 128:  # Pokud je pixel masky dostatečně světlý
                colored_mask.putpixel((x, y), color)
    
    # Překryjeme původní obrázek s barevnou maskou
    return Image.alpha_composite(img_rgba, colored_mask)

def generate_distinct_colors(n_colors, alpha=100):
    """
    Generuje n vizuálně odlišných barev v RGBA formátu.
    Používá HSV barevný model k rovnoměrnému rozdělení barevného spektra.
    
    Args:
        n_colors: Počet barev k vygenerování
        alpha: Hodnota průhlednosti (0-255)
        
    Returns:
        Seznam n barev v RGBA formátu
    """
    colors = []
    for i in range(n_colors):
        # Rovnoměrně rozdělí barevný kruh (H - Odstín), plná saturace (S - Sytost) a hodnota (V - Jas)
        h = i / n_colors
        s = 0.8  # Dostatečná saturace pro výrazné barvy
        v = 0.9  # Dostatečný jas pro viditelnost
        
        # Převod z HSV do RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        
        # Převod na 8-bitové hodnoty RGB a přidání alpha kanálu
        colors.append((int(r*255), int(g*255), int(b*255), alpha))
    
    return colors

def segment_image(image_array, hf_token):
    """
    Segmentuje obrázek pomocí Mask2Former modelu přes Hugging Face API
    Vrací obrázek se segmentačními maskami a unikátní třídy objektů
    """
    # Konverze numpy array na PIL Image
    pil_img = Image.fromarray(image_array)
    
    # Příprava obrázku pro API
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=90)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # API požadavek
    url = f"https://api-inference.huggingface.co/models/{SEGMENTATION_MODEL['id']}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    try:
        response = requests.post(url, headers=headers, json={"inputs": img_str}, timeout=30)
        if response.status_code == 200:
            results = response.json()
            labels = []
            
            # Připravíme výstupní obrázek v RGBA režimu pro překrytí masek
            output_image = pil_img.convert("RGBA")
            
            # Zjistíme počet segmentů a vygenerujeme unikátní barvy
            n_segments = len(results) if isinstance(results, list) else 0
            if n_segments > 0:
                # Vygenerovat unikátní barvy podle počtu segmentů
                colors = generate_distinct_colors(n_segments)
                
                for i, segment in enumerate(results):
                    if "label" in segment:
                        label = segment["label"].split(":")[-1].strip()
                        labels.append(label)
                    
                    # Zpracování Base64 kódované masky
                    if "mask" in segment:
                        try:
                            # Dekódování masky
                            mask_img = decode_base64_mask(segment["mask"])
                            if mask_img:
                                # Aplikování barevné masky s unikátní barvou
                                color = colors[i]
                                output_image = apply_colored_mask(output_image, mask_img, color)
                        except Exception as e:
                            st.warning(f"Chyba při zpracování masky: {str(e)}")
            
            # Převedeme zpět na RGB pro zobrazení
            return output_image.convert("RGB"), list(set(labels))
        
        st.error(f"Chyba API: {response.status_code}")
        return pil_img, []
    
    except Exception as e:
        st.error(f"Chyba při komunikaci s API: {str(e)}")
        return pil_img, []
