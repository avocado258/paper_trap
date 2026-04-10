import os
import csv

folders = [
    '../../data/fake_papers(1-84)',
    '../../data/fake_papers(85-250)'
]

results = []

for folder in folders:
    txt_files = [f for f in os.listdir(folder) if f.endswith('.txt')]
    print(f"\nFolder: {folder} — {len(txt_files)} txt files found")
    
    for i, filename in enumerate(txt_files):
        txt_path = os.path.join(folder, filename)
        
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
            
            if len(text) > 100:
                results.append({
                    'filename': filename,
                    'text': text,
                    'label': 'AI'
                })
                print(f"  Processed {i+1}/{len(txt_files)}: {filename}")
            else:
                print(f"  Too short, skipped: {filename}")
                
        except Exception as e:
            print(f"  Failed: {filename} — {e}")

# Save to CSV
with open('../../data/fake_papers.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['filename', 'text', 'label'])
    writer.writeheader()
    writer.writerows(results)

print(f"\nDone! {len(results)} fake papers saved to fake_papers.csv")