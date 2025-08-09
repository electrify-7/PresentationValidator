import json

def format_dict(data):
    formatted = []
    for slide_item in data.get('slide_titles', []):
        slide_num = slide_item['slide']
        title = slide_item['title']
        
        # Group / sort them
        texts = [t['text'] for t in data.get('texts', []) if t['slide'] == slide_num]
        tables = [tbl['table'] for tbl in data.get('tables', []) if tbl['slide'] == slide_num]
        charts = [c for c in data.get('charts', []) if c.get('slide') == slide_num]
        alt_texts = [a for a in data.get('alt_texts', []) if a.get('slide') == slide_num]
        images = [img for img in data.get('images', []) if img.get('slide') == slide_num]
        notes = [n for n in data.get('notes', []) if n.get('slide') == slide_num]
        
        formatted.append({
            "slide": slide_num,
            "title": title,
            "text_blocks": texts,
            "tables": tables,
            "charts": charts,
            "alt_texts": alt_texts,
            "images": images,
            "notes": notes
        })
    
    return formatted