# Translate all of the image data into description. OCR + captioning

from PIL import Image
from io import BytesIO
import pytesseract

def extract_text_from_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    text = pytesseract.image_to_string(image)
    return text.strip()

def caption(collected_data):
    for image_info in collected_data.get('images', []):
        if 'blob' in image_info and image_info['blob']:
            # Extract OCR text from the image blob
            ocr_text = extract_text_from_image(image_info['blob'])
            image_info['ocr_text'] = ocr_text
        else:
            image_info['ocr_text'] = ""
    return collected_data