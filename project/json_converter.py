def format_dict(data):
    formatted = []

    # Appending logic even incase it misses the titling/grouping

    slide_numbers = []
    slide_numbers += [s['slide'] for s in data.get('slide_titles', []) if 'slide' in s]
    slide_numbers += [t['slide'] for t in data.get('texts', []) if 'slide' in t]
    slide_numbers += [tbl['slide'] for tbl in data.get('tables', []) if 'slide' in tbl]
    slide_numbers += [c['slide'] for c in data.get('charts', []) if 'slide' in c]
    slide_numbers += [a['slide'] for a in data.get('alt_texts', []) if 'slide' in a]
    slide_numbers += [img['slide'] for img in data.get('images', []) if 'slide' in img]
    slide_numbers += [n['slide'] for n in data.get('notes', []) if 'slide' in n]
    slide_numbers += [s['slide'] for s in data.get('elements', []) if 'slide' in s]

    max_slide = max(slide_numbers) if slide_numbers else 0

    for slide_num in range(1, max_slide + 1):
        # ordering we made in pptx_parser (slide_element logic)

        el_entry = next((s for s in data.get('elements', []) if s.get('slide') == slide_num), None)
        if el_entry:
            elements = el_entry.get('elements', [])
            texts = [e['text'] for e in elements if e.get('type') == 'text' and e.get('text')]
            tables = [e['table'] for e in elements if e.get('type') == 'table']
            charts = [e.get('chart') for e in elements if e.get('type') == 'chart']
            alt_texts = [e.get('alt_text') for e in elements if 'alt_text' in e and e.get('alt_text')]
            images = [e for e in elements if e.get('type') == 'image']
            notes = [n['notes'] for n in data.get('notes', []) if n.get('slide') == slide_num]
            # title fallback from slide_titles
            title_entry = next((t for t in data.get('slide_titles', []) if t.get('slide') == slide_num), None)
            title = title_entry.get('title') if title_entry else None
        else:
            title_entry = next((t for t in data.get('slide_titles', []) if t.get('slide') == slide_num), None)
            title = title_entry.get('title') if title_entry else None
            texts = [t['text'] for t in data.get('texts', []) if t['slide'] == slide_num]
            tables = [tbl['table'] for tbl in data.get('tables', []) if tbl['slide'] == slide_num]
            charts = [c for c in data.get('charts', []) if c.get('slide') == slide_num]
            alt_texts = [a for a in data.get('alt_texts', []) if a.get('slide') == slide_num]
            images = [img for img in data.get('images', []) if img.get('slide') == slide_num]
            notes = [n['notes'] for n in data.get('notes', []) if n.get('slide') == slide_num]
        
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