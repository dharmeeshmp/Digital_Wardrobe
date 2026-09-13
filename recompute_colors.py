import os
import json
import sqlite3
import numpy as np

from main import DB_FILE, UPLOAD_DIR, analyze_dress_shade

conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute("SELECT id, filename FROM inventory")
items = c.fetchall()

print(f"Updating shades for {len(items)} stored inventory items...")

for item_id, filename in items:
    img_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(img_path):
        shade_name, dominant_lab = analyze_dress_shade(img_path)
        c.execute("""
            UPDATE inventory 
            SET shade_name = ?, dominant_lab = ? 
            WHERE id = ?
        """, (shade_name, json.dumps(dominant_lab.tolist()), item_id))
        print(f"  [OK] {filename} -> {shade_name}")

conn.commit()
conn.close()
print("\nInventory database successfully updated with the expanded color palette!")