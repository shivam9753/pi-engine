import csv
import json

def parse_csv_poems(filename, limit=50):
    """Parse CSV and extract individual poems"""
    poems = []
    
    with open(filename, 'r', encoding='utf-8-sig') as file:
        content = file.read()
        
        # Split by lines and try to identify poem boundaries
        lines = content.split('\n')
        
        current_poem = {}
        in_text = False
        text_lines = []
        poem_count = 0
        
        for i, line in enumerate(lines[1:], 1):  # Skip header
            if poem_count >= limit:
                break
                
            # Look for date pattern to identify new poems
            if '2024-' in line and 'T' in line and 'Z' in line:
                # Save previous poem if exists
                if current_poem and text_lines:
                    current_poem['text'] = '\n'.join(text_lines).strip()
                    poems.append(current_poem)
                    poem_count += 1
                
                # Start new poem
                parts = line.split(',', 2)
                if len(parts) >= 3:
                    current_poem = {
                        'id': '',
                        'added_date': parts[1] if len(parts) > 1 else '',
                        'author': parts[2].split(',')[0].strip('"') if len(parts) > 2 else ''
                    }
                    text_lines = []
                    in_text = False
            else:
                # Collect text lines
                if current_poem:
                    # Skip author bio lines and look for poem content
                    if not in_text and ('"""' in line or line.strip().startswith('1.')):
                        in_text = True
                    
                    if in_text and line.strip():
                        text_lines.append(line.strip())
        
        # Don't forget the last poem
        if current_poem and text_lines:
            current_poem['text'] = '\n'.join(text_lines).strip()
            poems.append(current_poem)
    
    return poems

# Parse first 10 poems as a test
poems = parse_csv_poems('C:/pi-engine/english_poetry.csv', 10)

print(f"Found {len(poems)} poems")
for i, poem in enumerate(poems[:3], 1):
    print(f"\n--- Poem {i} ---")
    print(f"Author: {poem['author']}")
    print(f"Date: {poem['added_date']}")
    print(f"Text preview: {poem['text'][:100]}...")