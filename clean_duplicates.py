import hashlib
import json
import os
import sqlite3
import numpy as np

DB_FILE = "inventory.db"
UPLOAD_DIR = "uploads"

def get_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash to detect bit-identical images."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_shade_similarity(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Perceptual color distance matching the main backend formula."""
    dL = lab1[0] - lab2[0]
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]
    dist = np.sqrt((0.35 * (dL ** 2)) + (1.35 * (da ** 2)) + (1.35 * (db ** 2)))
    score = 1.0 - (dist / 85.0)
    return float(np.clip(score, 0.0, 1.0))

def remove_duplicates():
    if not os.path.exists(DB_FILE):
        print("Database not found!")
        return

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("SELECT id, dress_name, filename, garment_type, shade_name, dominant_lab, pattern_emb FROM inventory")
    rows = c.fetchall()

    if not rows:
        print("Inventory is already empty.")
        conn.close()
        return

    print(f"Scanning {len(rows)} items for duplicates...")

    kept_items = []
    kept_hashes = set()
    ids_to_delete = []
    files_to_delete = []

    for row in rows:
        item_id = row[0]
        dress_name = row[1]
        filename = row[2]
        garment_type = row[3]
        shade_name = row[4]
        dominant_lab = np.array(json.loads(row[5]))
        pattern_emb = np.array(json.loads(row[6]))

        file_path = os.path.join(UPLOAD_DIR, filename)

        # 1. Flag missing files for removal
        if not os.path.exists(file_path):
            ids_to_delete.append(item_id)
            print(f"  [MISSING FILE] Removing stale record: {dress_name} ({filename})")
            continue

        # 2. Check exact SHA-256 match
        file_hash = get_file_sha256(file_path)
        if file_hash in kept_hashes:
            ids_to_delete.append(item_id)
            files_to_delete.append(file_path)
            print(f"  [EXACT DUPLICATE] Marked for deletion: {dress_name} ({filename})")
            continue

        # 3. Check visual duplicate (same category, high pattern and shade match)
        is_visual_dup = False
        for kept in kept_items:
            if kept["garment_type"].lower() != garment_type.lower():
                continue

            pattern_sim = float(np.dot(pattern_emb, kept["pattern_emb"]))
            if pattern_sim >= 0.985:
                shade_sim = compute_shade_similarity(dominant_lab, kept["dominant_lab"])
                if shade_sim >= 0.96:
                    is_visual_dup = True
                    ids_to_delete.append(item_id)
                    files_to_delete.append(file_path)
                    print(f"  [VISUAL DUPLICATE] '{dress_name}' is duplicate of '{kept['dress_name']}'")
                    break

        if not is_visual_dup:
            kept_hashes.add(file_hash)
            kept_items.append({
                "id": item_id,
                "dress_name": dress_name,
                "garment_type": garment_type,
                "dominant_lab": dominant_lab,
                "pattern_emb": pattern_emb
            })

    # Execute DB deletion
    if ids_to_delete:
        c.executemany("DELETE FROM inventory WHERE id = ?", [(i,) for i in ids_to_delete])
        conn.commit()

        # Delete image files from disk
        for fp in files_to_delete:
            try:
                if os.path.exists(fp):
                    os.remove(fp)
            except Exception as e:
                print(f"  [FILE ERROR] Could not remove {fp}: {e}")

    conn.close()

    print("\n--- Deduplication Summary ---")
    print(f"Original Count : {len(rows)}")
    print(f"Duplicates Cut : {len(ids_to_delete)}")
    print(f"Unique Items   : {len(kept_items)}")

if __name__ == "__main__":
    remove_duplicates()