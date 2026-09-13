import os
import urllib.request

SAMPLE_DIR = "sample_test_images"
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Direct links to diverse royalty-free apparel sample photos
SAMPLES = {
    # Pinks & Shades (Dresses & Kurtis)
    "baby_pink_summer_dress.jpg": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=500&q=80",
    "hot_pink_cocktail_dress.jpg": "https://images.unsplash.com/photo-1568252542512-9fe8fe9c87bb?w=500&q=80",
    "magenta_floral_dress.jpg": "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=500&q=80",

    # Men's Casual & Formal Wear
    "blue_denim_jacket.jpg": "https://images.unsplash.com/photo-1576995853123-5a10305d93c0?w=500&q=80",
    "black_formal_suit_blazer.jpg": "https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=500&q=80",
    "white_casual_tshirt.jpg": "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?w=500&q=80",
    "olive_green_hoodie.jpg": "https://images.unsplash.com/photo-1556905055-8f358a7a47b2?w=500&q=80",

    # Patterns & Prints
    "black_white_striped_shirt.jpg": "https://images.unsplash.com/photo-1523381210434-271e8be1f52b?w=500&q=80",
    "yellow_floral_print_dress.jpg": "https://images.unsplash.com/photo-1585487000160-6ebcfceb0d03?w=500&q=80",
    "red_plaid_check_shirt.jpg": "https://images.unsplash.com/photo-1589310243389-96a5483213a8?w=500&q=80",

    # Traditional & Bottoms
    "blue_denim_jeans.jpg": "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?w=500&q=80",
    "red_silk_ethnic_saree.jpg": "https://images.unsplash.com/photo-1610030469983-98e550d6193c?w=500&q=80",
}

print(f"Downloading {len(SAMPLES)} sample images into '{SAMPLE_DIR}/'...\n")

headers = {'User-Agent': 'Mozilla/5.0'}
for filename, url in SAMPLES.items():
    filepath = os.path.join(SAMPLE_DIR, filename)
    if not os.path.exists(filepath):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as resp, open(filepath, 'wb') as out_file:
                out_file.write(resp.read())
            print(f"  [DONE] {filename}")
        except Exception as e:
            print(f"  [ERROR] Could not download {filename}: {e}")
    else:
        print(f"  [EXISTS] {filename}")

print("\nAll sample images ready in 'sample_test_images' folder!")