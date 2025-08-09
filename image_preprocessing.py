"""
image_folder_parser.py

Convert a folder of slide images into the same collected_data dictionary format
produced by your pptx parser.

Dependencies:
  - opencv-python
  - numpy
  - pytesseract
"""

import os
import cv2
import re
import numpy as np
from pytesseract import Output
import pytesseract
from typing import List, Dict, Any

EXT_TO_CONTENT_TYPE = {
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg'
}

# -------------------------
# Reused / adapted helpers
# -------------------------

def deskew_image(gray):
    """Basic deskew via min area rectangle of text edges. Returns corrected grayscale and angle."""
    coords = np.column_stack(np.where(gray < 255))
    if coords.shape[0] < 10:
        return gray, 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    M = cv2.getRotationMatrix2D(rect[0], angle, 1.0)
    h, w = gray.shape
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle

def preprocess_image_cv(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot open {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img, thresh

def ocr_crop_text_with_conf(crop_bgr, tesseract_config="--oem 3 --psm 6"):
    """
    OCR a BGR crop (numpy array) with pytesseract and return:
      (text: str, avg_conf: float, raw_data: dict)

    - Groups words into lines using pytesseract's line numbers.
    - avg_conf is the mean of non-negative confidences, 0.0 if none.
    - raw_data is the dict returned by pytesseract.image_to_data for debug.
    - tesseract_config can be used to tune OCR (--psm, --oem, -l lang, etc).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0, {}

    # Convert BGR -> RGB for pytesseract
    try:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        # If crop is already single-channel or weird, try converting differently
        crop_rgb = crop_bgr

    # Run pytesseract (word-level data)
    data = pytesseract.image_to_data(crop_rgb, output_type=Output.DICT, config=tesseract_config)

    n = len(data.get('text', []))
    if n == 0:
        return "", 0.0, data

    # Group words into lines using pytesseract's line_num (fallback to y coords)
    lines = {}
    confs = []
    # safe fetch helpers
    line_nums = data.get('line_num', [0]*n)
    texts = data.get('text', ['']*n)
    conf_list = data.get('conf', ['-1']*n)

    for i in range(n):
        txt = str(texts[i]).strip()
        if not txt:
            continue
        # parse confidence safely
        try:
            conf = float(conf_list[i])
        except Exception:
            conf = -1.0
        if conf >= 0:
            confs.append(conf)

        ln = int(line_nums[i]) if line_nums and len(line_nums) > i else 0
        lines.setdefault(ln, []).append(txt)

    # Join lines in ascending order of line number
    joined_lines = [ " ".join(lines[k]) for k in sorted(lines.keys()) if lines.get(k) ]
    text = "\n".join(joined_lines).strip()

    avg_conf = float(np.mean(confs)) if confs else 0.0

    return text, avg_conf, data


def prepare_for_ocr(bgr_img, scale=2.0, do_deskew=True):
    """
    Upscale and convert to grayscale for better OCR. Returns processed bgr and gray.
    scale: multiply resolution to help OCR small fonts.
    """
    if scale != 1.0:
        h, w = bgr_img.shape[:2]
        bgr_img = cv2.resize(bgr_img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    if do_deskew:
        gray, _ = deskew_image(gray)
    # optional denoise/CLAHE
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    return bgr_img, gray

def ocr_words_from_image(bgr_img):
    """
    Run pytesseract on image and return list of words with boxes and confidences.
    Each item: dict {text, left, top, width, height, conf, line_num, block_num}
    Coordinates are in image pixel space.
    """
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb, output_type=Output.DICT)
    n = len(data['text'])
    words = []
    for i in range(n):
        txt = data['text'][i].strip()
        if not txt:
            continue
        try:
            conf = float(data['conf'][i])
        except:
            conf = -1.0
        words.append({
            'text': txt,
            'left': int(data['left'][i]),
            'top': int(data['top'][i]),
            'width': int(data['width'][i]),
            'height': int(data['height'][i]),
            'conf': conf,
            'line_num': int(data.get('line_num', [0]*n)[i]),
            'block_num': int(data.get('block_num', [0]*n)[i])
        })
    return words

def group_words_to_lines(words, y_tol=0.6):
    """
    Group word boxes into lines using word vertical centers.
    y_tol: fraction of median line height used to decide same-line grouping.
    Returns list of lines; each line is list of words (sorted by left).
    """
    if not words:
        return []

    # compute word center y and heights
    for w in words:
        w['cy'] = w['top'] + w['height'] / 2.0

    heights = [w['height'] for w in words]
    median_h = float(np.median(heights)) if heights else 10.0
    tol_px = max(3, median_h * y_tol)

    # sort by cy then left
    words_sorted = sorted(words, key=lambda w: (w['cy'], w['left']))

    lines = []
    current_line = [words_sorted[0]]
    for w in words_sorted[1:]:
        if abs(w['cy'] - np.median([x['cy'] for x in current_line])) <= tol_px:
            current_line.append(w)
        else:
            # finish current line
            current_line = sorted(current_line, key=lambda x: x['left'])
            lines.append(current_line)
            current_line = [w]
    # append last
    if current_line:
        current_line = sorted(current_line, key=lambda x: x['left'])
        lines.append(current_line)

    # build line metadata (y_top, y_bottom, avg_conf)
    out_lines = []
    for ln in lines:
        left = min(w['left'] for w in ln)
        right = max(w['left'] + w['width'] for w in ln)
        top = min(w['top'] for w in ln)
        bottom = max(w['top'] + w['height'] for w in ln)
        text = " ".join([w['text'] for w in ln])
        avg_conf = float(np.mean([w['conf'] for w in ln if w['conf'] >= 0])) if any(w['conf'] >= 0 for w in ln) else 0.0
        out_lines.append({'text': text, 'left': left, 'top': top, 'width': right-left, 'height': bottom-top, 'conf': avg_conf})
    return out_lines

def merge_lines_to_blocks(lines, vert_gap_mult=1.5, min_block_width=40):
    """
    Merge adjacent lines into blocks based on vertical gap and x-overlap.
    vert_gap_mult: threshold multiplier of median line height to consider a paragraph break.
    Returns list of blocks: {'text': ..., 'bbox': [x,y,w,h], 'conf': avg_conf}
    """
    if not lines:
        return []
    heights = [l['height'] for l in lines]
    median_h = float(np.median(heights)) if heights else 10.0
    max_gap = median_h * vert_gap_mult

    blocks = []
    current_block = [lines[0]]
    for prev, curr in zip(lines, lines[1:]):
        gap = curr['top'] - (prev['top'] + prev['height'])
        # check x-overlap: if their x ranges align reasonably, treat as same paragraph
        prev_x0, prev_x1 = prev['left'], prev['left'] + prev['width']
        curr_x0, curr_x1 = curr['left'], curr['left'] + curr['width']
        overlap = min(prev_x1, curr_x1) - max(prev_x0, curr_x0)
        min_width = min(prev['width'], curr['width'])
        x_align = (overlap > 0.2 * min_width)  # 20% overlap threshold

        if gap <= max_gap and x_align:
            current_block.append(curr)
        else:
            # finish block
            blocks.append(current_block)
            current_block = [curr]
    if current_block:
        blocks.append(current_block)

    # convert blocks to single text + bbox
    out_blocks = []
    for blk in blocks:
        x0 = min(l['left'] for l in blk)
        x1 = max(l['left'] + l['width'] for l in blk)
        y0 = min(l['top'] for l in blk)
        y1 = max(l['top'] + l['height'] for l in blk)
        text = "\n".join([l['text'] for l in blk])
        confs = [l['conf'] for l in blk if l['conf'] is not None]
        avg_conf = float(np.mean(confs)) if confs else 0.0
        if (x1 - x0) < min_block_width:
            # ignore super small junk
            continue
        out_blocks.append({'text': text, 'bbox': [int(x0), int(y0), int(x1 - x0), int(y1 - y0)], 'conf': avg_conf})
    return out_blocks

# Convenience wrapper: image -> blocks
def image_to_text_blocks(bgr_img, scale=2.0, deskew=True):
    bgr, gray = prepare_for_ocr(bgr_img, scale=scale, do_deskew=deskew)
    words = ocr_words_from_image(bgr)
    lines = group_words_to_lines(words, y_tol=0.6)
    blocks = merge_lines_to_blocks(lines, vert_gap_mult=1.6)
    # If scale != 1, scale down bbox coords to original image size factors:
    if scale != 1.0:
        h_scale = 1.0 / scale
        for b in blocks:
            x,y,w,h = b['bbox']
            b['bbox'] = [int(x * h_scale), int(y * h_scale), int(w * h_scale), int(h * h_scale)]
    return blocks


def clamp_bbox(bbox, W, H):
    x,y,w,h = bbox
    x = max(0, int(x)); y = max(0, int(y))
    w = max(1, int(min(w, W-x))); h = max(1, int(min(h, H-y)))
    return x,y,w,h


# ---------- Simple table detection/parse (heuristic) ----------
def detect_table_grid(crop_gray_bin):
    img = crop_gray_bin.copy()
    horiz_size = max(8, img.shape[1] // 30)
    vert_size = max(8, img.shape[0] // 30)
    horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_size, 1))
    vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_size))

    er_h = cv2.erode(img, horiz)
    dil_h = cv2.dilate(er_h, horiz)

    er_v = cv2.erode(img, vert)
    dil_v = cv2.dilate(er_v, vert)

    mask = cv2.bitwise_and(dil_h, dil_v)
    if cv2.countNonZero(mask) < 800:  # cheap threshold if not table-like
        return None
    return mask

def extract_table_cells_from_mask(table_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.dilate(table_mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w > 10 and h > 8 and cv2.contourArea(c) > 80:
            cells.append((x,y,w,h))
    cells = sorted(cells, key=lambda b: (b[1], b[0]))
    # deduplicate heavy overlaps
    filtered = []
    for (x,y,w,h) in cells:
        keep = True
        for (fx,fy,fw,fh) in filtered:
            inter = max(0, min(x+w, fx+fw)-max(x,fx)) * max(0, min(y+h, fy+fh)-max(y,fy))
            if inter > 0.6 * (w*h):
                keep = False
                break
        if keep:
            filtered.append((x,y,w,h))
    return filtered

def group_cells_to_grid(cells, tol=10):
    if not cells: return []
    rows = []
    for c in cells:
        x,y,w,h = c
        placed = False
        for r in rows:
            rx,ry,rw,rh = r[0]
            if abs(y - ry) <= tol:
                r.append(c); placed = True; break
        if not placed:
            rows.append([c])
    grid = [sorted(r, key=lambda b: b[0]) for r in rows]
    return grid

def parse_table_from_crop(crop_bgr, crop_bin):
    """
    Improved table parser:
    - Detects horizontal & vertical lines via morphology
    - Computes intersections to infer grid x/y coordinates
    - Builds cell rectangles from adjacent x/y coords and OCRs each cell
    - If grid inference fails, falls back to simple vertical-projection column detection
    Returns: table as list-of-rows (strings) or None if no table-like structure found.
    """
    h, w = crop_bin.shape[:2]

    # --- 1) Detect horizontal and vertical lines (adaptive sizes) ---
    horiz_size = max(8, w // 40)
    vert_size  = max(8, h // 40)

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_size, 1))
    vert_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_size))

    horiz = cv2.erode(crop_bin, horiz_kernel, iterations=1)
    horiz = cv2.dilate(horiz, horiz_kernel, iterations=1)

    vert = cv2.erode(crop_bin, vert_kernel, iterations=1)
    vert = cv2.dilate(vert, vert_kernel, iterations=1)

    # Clean small noise
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_OPEN, kernel_small)
    vert  = cv2.morphologyEx(vert, cv2.MORPH_OPEN, kernel_small)

    # intersection points
    inter = cv2.bitwise_and(horiz, vert)
    if cv2.countNonZero(inter) < 50:
        # not enough intersections — try fallback (vertical projection)
        return _parse_table_fallback_vproj(crop_bgr, crop_bin)

    # --- 2) collect intersection centers and cluster unique x / y coords ---
    ys, xs = np.where(inter > 0)
    pts = np.column_stack((xs, ys))  # (x,y)

    # cluster by rounding into bins to consolidate near intersections
    # choose bin sizes relative to image
    bin_x = max(6, w // 200)
    bin_y = max(6, h // 200)

    # create dict of bins -> mean coordinate
    from collections import defaultdict
    bins_x = defaultdict(list)
    bins_y = defaultdict(list)
    for xpt, ypt in pts:
        bx = int(round(xpt / bin_x))
        by = int(round(ypt / bin_y))
        bins_x[bx].append(xpt)
        bins_y[by].append(ypt)

    xs_unique = sorted([int(np.mean(v)) for v in bins_x.values()])
    ys_unique = sorted([int(np.mean(v)) for v in bins_y.values()])

    # need at least 2 vertical and 2 horizontal grid lines to form cells
    if len(xs_unique) < 2 or len(ys_unique) < 2:
        return _parse_table_fallback_vproj(crop_bgr, crop_bin)

    # --- 3) build grid from adjacent unique coords ---
    # Note: intersections correspond to gridlines; cell boundaries are between adjacent lines
    table = []
    for r in range(len(ys_unique)-1):
        row_cells = []
        y0 = ys_unique[r]
        y1 = ys_unique[r+1]
        # expand vertical limits slightly to include text near lines
        y0c = max(0, y0)
        y1c = min(h, y1)
        for c in range(len(xs_unique)-1):
            x0 = xs_unique[c]
            x1 = xs_unique[c+1]
            x0c = max(0, x0)
            x1c = min(w, x1)
            # add small padding to avoid cutting characters near border
            pad = 3
            x0p = max(0, x0c + pad)
            y0p = max(0, y0c + pad)
            x1p = min(w, x1c - pad)
            y1p = min(h, y1c - pad)
            # sometimes coords can cross; ensure positive width/height
            if x1p <= x0p or y1p <= y0p:
                row_cells.append("")  # empty cell
                continue
            cell_crop = crop_bgr[y0p:y1p, x0p:x1p]
            txt, conf, _ = ocr_crop_text_with_conf(cell_crop)
            row_cells.append(txt.strip())
        table.append(row_cells)

    # validation: table should have >1 column and >0 non-empty cells
    if not table or max(len(r) for r in table) < 2:
        return _parse_table_fallback_vproj(crop_bgr, crop_bin)

    non_empty = sum(1 for r in table for c in r if c and c.strip())
    if non_empty == 0:
        return _parse_table_fallback_vproj(crop_bgr, crop_bin)

    return table


def _parse_table_fallback_vproj(crop_bgr, crop_bin):
    """
    Lightweight fallback: try to detect columns by vertical projection of text pixels.
    This helps for tables that lack clear grid lines (common in screenshots).
    Returns table-like rows by slicing image vertically then OCR each slice and splitting lines.
    """
    h, w = crop_bin.shape[:2]
    # invert bin so text pixels are white (if needed)
    inv = 255 - crop_bin

    # vertical projection
    vproj = np.sum(inv > 0, axis=0)  # counts of text pixels per column
    # smooth projection and find valleys -> split columns
    win = max(3, w // 200)
    vproj_s = np.convolve(vproj, np.ones(win)/win, mode='same')

    # find potential column separators where projection is low
    thresh = max(1, int(np.max(vproj_s) * 0.05))
    separators = np.where(vproj_s < thresh)[0]

    # find continuous ranges of separators and pick split points between columns
    splits = []
    if len(separators) == 0:
        # cannot find separators - treat whole crop as single column -> try to extract lines
        txt_full, conf_full, _ = ocr_crop_text_with_conf(crop_bgr)
        if not txt_full.strip():
            return None
        # convert text lines to a single-row table where each line is a row in 1 column
        rows = [[ln.strip()] for ln in txt_full.splitlines() if ln.strip()]
        return rows if rows else None

    # find boundaries of separator runs
    groups = np.split(separators, np.where(np.diff(separators) != 1)[0] + 1)
    # choose split x positions between groups
    cut_x = []
    last = 0
    for g in groups:
        xstart = int(g[0])
        xend = int(g[-1])
        # column ends at the midpoint between last and start of this separator
        mid = int(max(last, (last + xstart)//2))
        cut_x.append(mid)
        last = xend
    # final column right boundary
    cut_x.append(w)
    # build column x ranges from previous cut points
    col_ranges = []
    left = 0
    for right in cut_x:
        if right - left > 10:
            col_ranges.append((left, right))
        left = right+1

    # OCR each column and produce rows by line alignment (best-effort)
    column_texts = []
    for (x0, x1) in col_ranges:
        x0c = max(0, x0); x1c = min(w, x1)
        col_crop = crop_bgr[:, x0c:x1c]
        txt, conf, _ = ocr_crop_text_with_conf(col_crop)
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        column_texts.append(lines)

    # transpose columns into rows (shortest column limits row count)
    if not column_texts:
        return None
    max_rows = max(len(c) for c in column_texts)
    table = []
    for r in range(max_rows):
        row = []
        for c in column_texts:
            row.append(c[r] if r < len(c) else "")
        table.append(row)
    # sanity check
    if len(table) >= 1 and max(len(r) for r in table) >= 2:
        return table
    return None
# -------------------------
# Higher-level heuristics
# -------------------------
def detect_possible_chart_from_text(text: str) -> bool:
    # crude: if OCR'd crop has axis-like words or multiple numeric tokens -> likely chart
    numeric_tokens = re.findall(r"[-+]?\d[\d,\.%]*", text)
    axis_words = ['axis','x-axis','y-axis','year','value','percent','%','legend','series']
    if len(numeric_tokens) >= 2:
        return True
    lower = text.lower()
    if any(w in lower for w in axis_words):
        return True
    return False

def crop_to_blob_bytes(crop_bgr, ext='png'):
    ok, buf = cv2.imencode(f".{ext}", crop_bgr)
    if not ok:
        return None
    return buf.tobytes()

# -------------------------
# Main: produce collected_data
# -------------------------
def parse_image_folder(folder_path: str, save_crops: bool = False) -> Dict[str, Any]:
    """
    Returns collected_data with keys identical to your PPTX parser:
    {
      'file': folder_path,
      'slide_titles': [{'slide': i, 'title': ...}, ...],
      'texts': [{'slide': i, 'text': ...}, ...],
      'alt_texts': [{'slide': i, 'alt_text': ...}, ...],
      'tables': [{'slide': i, 'table': [...]} , ... ],
      'images': [{'slide':i, 'filename':..., 'blob':..., 'ext':..., 'content_type':..., 'shape_id':... , 'alt_text': optional}, ...],
      'charts': [{'slide': i, 'title':..., 'series': [...], 'empty': True/False}, ...],
      'notes': []  # not available -> keep empty
    }
    """
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    collected_data = {
        'file': folder_path,
        'slide_titles': [],
        'texts': [],
        'alt_texts': [],
        'tables': [],
        'images': [],
        'charts': [],
        'notes': []
    }

    for i, fn in enumerate(files, start=1):
        path = os.path.join(folder_path, fn)
        try:
            color_img, thresh = preprocess_image_cv(path)
        except Exception as e:
            print(f"[parse_image_folder] cannot open {path}: {e}")
            # still include a placeholder slide_title
            collected_data['slide_titles'].append({'slide': i, 'title': None})
            continue

        H, W = thresh.shape
        #bboxes = image_to_text_blocks(color_img,scale=2.0, deskew=True)

        # simple heuristic for slide title: look for top-most wide short box
        ocr_scale = 2.0  # tune this: 1.0 -> faster, 2.0 -> better small-font OCR
        blocks = image_to_text_blocks(color_img, scale=ocr_scale, deskew=True)

        # find a title candidate from the top-most wide, short block (prefer OCR blocks)
        title_text = None
        for blk in sorted(blocks, key=lambda b: (b['bbox'][1], b['bbox'][0])):  # top->bottom
            bx, by, bw, bh = blk['bbox']
            if bh < 80 and bw > 0.5 * W and by < 0.3 * H:
                if blk.get('text'):
                    title_text = blk['text'].strip()
                    break
        collected_data['slide_titles'].append({'slide': i, 'title': title_text if title_text else None})

        # process each OCR block (shape-like records)
        shape_id_counter = 1
        for blk in blocks:
            x, y, w, h = clamp_bbox(tuple(blk['bbox']), W, H)
            # ensure minimum size crop
            crop_bgr = color_img[y:y+h, x:x+w].copy()
            if crop_bgr.size == 0:
                continue
            crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            _, crop_bin = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # use existing heuristics but informed by OCR block content
            block_type = "text"
            if h < 60 and w > 0.5 * W:
                block_type = "title"
            elif w > 160 and h > 140:
                block_type = "table_or_figure"

            # prefer OCR text from block if available
            blk_text = blk.get('text', '').strip()
            blk_conf = blk.get('conf', 0.0)

            if block_type == "table_or_figure":
                # first try table parsing on the crop
                table = parse_table_from_crop(crop_bgr, crop_bin)
                if table:
                    collected_data['tables'].append({'slide': i, 'table': table})
                    tconcat = " | ".join([", ".join(row) for row in table])
                    collected_data['texts'].append({'slide': i, 'text': tconcat})
                else:
                    # not a grid table -> treat as figure/chart
                    # use OCR text from the block as caption candidate
                    txt = blk_text
                    conf = blk_conf
                    is_chart = detect_possible_chart_from_text(txt if txt else "")
                    chart_title = txt.splitlines()[0].strip() if txt else None
                    chart_entry = {
                        'slide': i,
                        'title': chart_title,
                        'series': [],                 # left empty: chart not digitized here
                        'empty': True if not is_chart else False
                    }
                    collected_data['charts'].append(chart_entry)

                    # alt_text fallback from OCR block
                    collected_data['alt_texts'].append({'slide': i, 'alt_text': txt if txt else None})

                    # create image blob for downstream (mimic PPTX image entry)
                    ext = 'png'
                    blob = crop_to_blob_bytes(crop_bgr, ext=ext)
                    filename = f"{os.path.splitext(fn)[0]}_shape{shape_id_counter}.{ext}"
                    content_type = EXT_TO_CONTENT_TYPE.get(ext, 'application/octet-stream')
                    image_info = {
                        'slide': i,
                        'filename': filename,
                        'blob': blob,
                        'ext': ext,
                        'content_type': content_type,
                        'shape_id': shape_id_counter
                    }
                    if txt:
                        image_info['alt_text'] = txt
                    collected_data['images'].append(image_info)

                    # also add OCR fallback text for LLM
                    if txt:
                        collected_data['texts'].append({'slide': i, 'text': txt})

            else:
                # text/title block -> use the combined OCR block text
                if blk_text:
                    collected_data['texts'].append({'slide': i, 'text': blk_text})
                    if block_type == 'title':
                        collected_data['alt_texts'].append({'slide': i, 'alt_text': blk_text})

            # optionally save crops for debugging
            if save_crops:
                dbg = os.path.join("debug_crops")
                os.makedirs(dbg, exist_ok=True)
                cname = os.path.join(dbg, f"{os.path.splitext(fn)[0]}_shape{shape_id_counter}.png")
                cv2.imwrite(cname, crop_bgr)

            shape_id_counter += 1

        # notes aren't present in flat images -> keep empty (same as PPTX when not present)

    return collected_data

# simple helper for converting crop to bytes:
def crop_to_blob_bytes(crop_bgr, ext='png'):
    ok, buf = cv2.imencode(f".{ext}", crop_bgr)
    if not ok:
        return None
    return buf.tobytes()

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    folder = "NoogatAssignment"
    out = parse_image_folder(folder, save_crops=True)
    import json
    # safe-serialize blobs as base64 if you want; here we omit (blobs are binary)
    # for demonstration we will write a JSON with placeholders for blobs
    serializable = dict(out)
    # convert blob to placeholder or base64 as needed
    for img in serializable['images']:
        img['blob'] = '<BINARY_IMAGE_BYTES>' if img['blob'] is not None else None
    with open("slides_as_collected_data.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print("Wrote slides_as_collected_data.json (blobs redacted in JSON file)")

