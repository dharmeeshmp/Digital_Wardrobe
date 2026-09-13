import os
import sqlite3

DB_FILE = "inventory.db"
UPLOAD_DIR = "uploads"

# 1. Clear database entries
if os.path.exists(DB_FILE):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM inventory")
    conn.commit()
    conn.close()
    print("Database cleared.")

# 2. Remove files inside uploads directory
if os.path.exists(UPLOAD_DIR):
    deleted_count = 0
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            deleted_count += 1
    print(f"Deleted {deleted_count} image files from {UPLOAD_DIR}/")

print("Inventory images and database wiped successfully!")