import os
import urllib.request
import time

TARGET_DIR = "sample_test_images"
os.makedirs(TARGET_DIR, exist_ok=True)

# Curated catalog covering colors, patterns, and clothing families
APPAREL_URLS = {
    # --- Tops & Shirts ---
    "white_crewneck_tshirt.jpg": "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?w=700&q=80",
    "black_cotton_tshirt.jpg": "https://images.unsplash.com/photo-1503342217505-b0a15ec3261c?w=700&q=80",
    "navy_blue_casual_shirt.jpg": "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=700&q=80",
    "red_plaid_flannel_shirt.jpg": "https://images.unsplash.com/photo-1589310243389-96a5483213a8?w=700&q=80",
    "black_white_striped_tshirt.jpg": "https://images.unsplash.com/photo-1523381210434-271e8be1f52b?w=700&q=80",
    "grey_hoodie_sweatshirt.jpg": "https://images.unsplash.com/photo-1556905055-8f358a7a47b2?w=700&q=80",
    "olive_green_crew_sweater.jpg": "https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=700&q=80",

    # --- Dresses & Gowns ---
    "baby_pink_summer_dress.jpg": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=700&q=80",
    "hot_pink_cocktail_dress.jpg": "https://images.unsplash.com/photo-1568252542512-9fe8fe9c87bb?w=700&q=80",
    "yellow_floral_maxi_dress.jpg": "https://images.unsplash.com/photo-1585487000160-6ebcfceb0d03?w=700&q=80",
    "black_evening_gown.jpg": "https://images.unsplash.com/photo-1566174053879-31528523f8ae?w=700&q=80",
    "red_formal_party_dress.jpg": "https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=700&q=80",
    "emerald_green_midi_dress.jpg": "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=700&q=80",
    "lavender_casual_sundress.jpg": "https://images.unsplash.com/photo-1515372039744-b8f02a3ae446?w=700&q=80",

    # --- Pants & Bottoms ---
    "blue_denim_jeans.jpg": "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?w=700&q=80",
    "black_slim_fit_jeans.jpg": "https://images.unsplash.com/photo-1584370848010-d7fe6bc767ec?w=700&q=80",
    "khaki_chino_trousers.jpg": "https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?w=700&q=80",
    "grey_formal_slacks.jpg": "https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=700&q=80",
    "denim_casual_shorts.jpg": "https://images.unsplash.com/photo-1591195853828-11db59a44f6b?w=700&q=80",
    "black_pleated_skirt.jpg": "https://images.unsplash.com/photo-1583496661160-fb5886a0aaaa?w=700&q=80",

    # --- Outerwear & Jackets ---
    "classic_blue_denim_jacket.jpg": "https://images.unsplash.com/photo-1576995853123-5a10305d93c0?w=700&q=80",
    "black_leather_biker_jacket.jpg": "https://images.unsplash.com/photo-1520975954732-35dd22299614?w=700&q=80",
    "navy_formal_suit_blazer.jpg": "https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=700&q=80",
    "beige_trench_coat.jpg": "https://images.unsplash.com/photo-1544441893-675973e31985?w=700&q=80",
    "olive_bomber_jacket.jpg": "https://images.unsplash.com/photo-1548883354-7622d03aca27?w=700&q=80",

    # --- Ethnic & Traditional ---
    "red_silk_kanchipuram_saree.jpg": "https://images.unsplash.com/photo-1610030469983-98e550d6193c?w=700&q=80",
    "maroon_embroidered_kurta.jpg": "https://images.unsplash.com/photo-1617627143750-d86bc21e42bb?w=700&q=80",
    "yellow_silk_wedding_saree.jpg": "https://images.unsplash.com/photo-1617196034183-421b4917c92d?w=700&q=80",
    "white_cotton_ethnic_kurti.jpg": "https://images.unsplash.com/photo-1583391733956-3750e0ff4e8b?w=700&q=80",
    "royal_blue_designer_lehenga.jpg": "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=700&q=80"
}

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

print(f"Starting bulk download of {len(APPAREL_URLS)} clothing images into '{TARGET_DIR}/'...\n")

downloaded, skipped, failed = 0, 0, 0

for filename, url in APPAREL_URLS.items():
    file_path = os.path.join(TARGET_DIR, filename)
    if os.path.exists(file_path):
        print(f"  [EXISTS] {filename}")
        skipped += 1
        continue

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp, open(file_path, "wb") as f:
            f.write(resp.read())
        print(f"  [DONE] {filename}")
        downloaded += 1
        time.sleep(0.3)
    except Exception as e:
        print(f"  [FAILED] {filename}: {e}")
        failed += 1

print("\n--- Summary ---")
print(f"Downloaded: {downloaded} | Skipped: {skipped} | Failed: {failed}")
print(f"Target Directory: {os.path.abspath(TARGET_DIR)}")