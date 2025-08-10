import os
import hashlib
from PIL import Image, ImageOps, ImageFilter
import pytesseract
from pytesseract import Output

# OCR-ing the image:

def _preprocess_pil_for_ocr(im: Image.Image, max_dim=2000) -> Image.Image:
    im = im.convert("RGB")
    # resize
    if max(im.size) > max_dim:
        scale = max_dim / max(im.size)
        new_size = (int(im.width * scale), int(im.height * scale))
        im = im.resize(new_size, Image.LANCZOS)
    
    im = im.filter(ImageFilter.MedianFilter(size=3))
    im = ImageOps.autocontrast(im)
    return im


def _ocr_image_to_text_and_blocks(pil_image: Image.Image, tesseract_config="--psm 3"):
    try:
        full_text = pytesseract.image_to_string(pil_image, config=tesseract_config).strip()
    except Exception:
        full_text = ""

    try:
        data = pytesseract.image_to_data(pil_image, config=tesseract_config, output_type=Output.DICT)
        # block_num -> par_num -> line_num // trying to have it grouped together :( 
        n = len(data['level'])
        lines = []
        last_key = None
        line_words = []
        for i in range(n):
            key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
            word = data['text'][i].strip()
            conf = int(data['conf'][i]) if data['conf'][i].isdigit() else -1
            if key != last_key and line_words:
                lines.append(" ".join(line_words))
                line_words = []
            if word:
                line_words.append(word)
            last_key = key
        if line_words:
            lines.append(" ".join(line_words))
    except Exception:
        lines = [] 

    return full_text, lines


def process_folder(path,do_ocr=True,image_extensions=('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
    collected_data = {
        'file': path,
        'slide_titles': [],  
        'texts': [],
        'alt_texts': [],
        'tables': [],         # no tables but well wouldn't hurt the rest of the pipeline to maintain same dict
        'images': [],
        'charts': [],
        'notes': [],
        'elements': []
    }

    if os.path.isdir(path):
        files = sorted([f for f in os.listdir(path) if f.lower().endswith(image_extensions)])
        if not files:
            raise ValueError("No supported image files found in the folder.")
        file_paths = [os.path.join(path, f) for f in files]
    elif os.path.isfile(path) and path.lower().endswith(image_extensions):
        file_paths = [path]
    else:
        raise ValueError(f"Invalid path!: {path}")

    for idx, file_path in enumerate(file_paths, start=1):
        fname = os.path.basename(file_path)

        image_info = {
            'slide': idx,
            'filename': fname,
            'orig_path': file_path,
            'ext': os.path.splitext(fname)[1].lstrip('.').lower() or None,
            'orig_filename': fname
        }

        ocr_text = ""
        text_blocks = []
        width = height = None

        try:
            with Image.open(file_path) as pil:
                width, height = pil.size
                if do_ocr:
                    pre = _preprocess_pil_for_ocr(pil)
                    ocr_text, text_blocks = _ocr_image_to_text_and_blocks(pre)
        except Exception as e:
            print(f"Error opening/OCR {file_path}: {e}")

        if ocr_text:
            image_info['ocr_text'] = ocr_text
            collected_data['texts'].append({'slide': idx, 'text': ocr_text})
        if text_blocks:
            image_info['text_blocks'] = text_blocks
        if width and height:
            image_info['width'] = width
            image_info['height'] = height

        collected_data['images'].append(image_info)
        collected_data['slide_titles'].append({'slide': idx, 'title': None})

        el = {
            'type': 'image',
            'slide': idx,
            'filename': fname,
            'orig_path': file_path,
            'ext': image_info['ext'],
            'shape_name': None
        }
        if ocr_text:
            el['ocr_text'] = ocr_text
        if text_blocks:
            el['text_blocks'] = text_blocks

        collected_data['elements'].append({'slide': idx, 'elements': [el]})

    return collected_data