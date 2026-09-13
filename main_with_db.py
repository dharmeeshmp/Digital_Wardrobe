import os
import sys
import uuid
import json
import shutil
import hashlib
import traceback
import time
import logging
from typing import List, Tuple

# Third-party libraries
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import torch
from transformers import CLIPModel, CLIPProcessor
from rembg import new_session, remove

# Web framework & ORM
from fastapi import FastAPI, File, Form, UploadFile, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

# Local database modules
from database import GarmentItem, init_db, get_db

# -------------------------------------------------------------
# 0. Dual Logging: Terminal Console + Persistent 'app.log' File
# -------------------------------------------------------------
LOG_FILE = "app.log"
handlers = [
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=handlers
)
logger = logging.getLogger("ApparelMatcher")

logger.info("Initializing database schema...")
init_db()
logger.info("Database schema initialized successfully.")

logger.info("Creating persistent U2-Net session ('u2netp')...")
rembg_session = new_session("u2netp")
logger.info("U2-Net session loaded into memory.")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Universal Apparel Matcher")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# -------------------------------------------------------------
# 1. Model Initialization
# -------------------------------------------------------------
models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models["device"] = device

logger.info("Loading CLIP model directly onto target device: %s", device)
t0 = time.perf_counter()
models["clip_model"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
models["clip_processor"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
logger.info("CLIP models loaded successfully in %.2f seconds.\n", time.perf_counter() - t0)

# -------------------------------------------------------------
# 2. Taxonomies & Palettes
# -------------------------------------------------------------
ALL_PATTERNS = [
    "solid plain color", "gingham check", "houndstooth pattern", "plaid or tartan",
    "glen plaid or prince of wales check", "harlequin diamond pattern", "buffalo check",
    "windowpane check", "argyle diamond pattern", "polka dot", "vertical stripes or pinstripe",
    "horizontal stripes", "awning wide stripes", "chevron or zig-zag pattern",
    "herringbone weave pattern", "greek key pattern", "trellis geometric pattern",
    "quatrefoil pattern", "floral print", "ditsy small floral print", "paisley pattern",
    "toile de jouy pattern", "chintz floral pattern", "damask ornate pattern",
    "jacquard or brocade weave", "fleur-de-lis pattern", "ikat pattern", "batik pattern",
    "tie-dye or shibori", "camouflage pattern", "ombre gradient pattern",
    "abstract graphic print", "geometric print", "leopard print", "cheetah print",
    "zebra stripe print", "snake or python skin print", "tiger stripe print",
    "cow print", "tortoiseshell pattern"
]

ALL_GARMENT_TYPES = [
    "casual shirt", "formal dress shirt", "t-shirt", "polo shirt",
    "tank top or vest", "hoodie or sweatshirt", "sweater or cardigan",
    "kurta or kurti", "sherwani", "saree", "salwar kameez",
    "lehenga choli", "dhoti or veshti or lungi",
    "jeans or denim pants", "formal trousers or slacks", "chinos or casual trousers",
    "cargo pants", "shorts", "skirt", "track pants or sweatpants",
    "suit or blazer", "jacket or coat", "leather jacket",
    "raincoat or windbreaker", "maxi dress or long gown",
    "cocktail or party dress", "casual summer dress", "jumpsuit or romper"
]

COLOR_PALETTE = {
    "Pure White": np.array([255, 128, 128]),
    "Off-White / Pearl": np.array([242, 127, 131]),
    "Ivory / Cream": np.array([238, 126, 142]),
    "Silver / Light Grey": np.array([215, 128, 128]),
    "Heather Grey": np.array([145, 128, 128]),
    "Charcoal / Dark Grey": np.array([75, 128, 128]),
    "Jet Black": np.array([20, 128, 128]),
    "Baby Pink / Light Pink": np.array([210, 150, 134]),
    "Rose Pink": np.array([170, 162, 137]),
    "Hot Pink": np.array([150, 202, 138]),
    "Fuchsia / Dark Pink": np.array([130, 192, 140]),
    "Magenta / Berry": np.array([105, 186, 146]),
    "Lavender / Lilac": np.array([195, 142, 116]),
    "Deep Purple / Plum": np.array([85, 165, 96]),
    "Bright Red": np.array([135, 200, 168]),
    "Maroon / Wine": np.array([75, 170, 145]),
    "Peach / Coral": np.array([195, 148, 158]),
    "Orange": np.array([170, 155, 185]),
    "Rust / Terracotta": np.array([120, 155, 165]),
    "Lemon Yellow": np.array([230, 120, 185]),
    "Mustard Yellow": np.array([165, 135, 180]),
    "Beige / Khaki": np.array([220, 130, 140]),
    "Tan / Camel": np.array([155, 136, 152]),
    "Chocolate Brown": np.array([70, 138, 142]),
    "Sky Blue": np.array([205, 118, 114]),
    "Teal / Turquoise": np.array([140, 102, 118]),
    "Royal Blue": np.array([95, 140, 80]),
    "Navy Blue": np.array([45, 134, 98]),
    "Mint Green": np.array([205, 108, 136]),
    "Olive Green": np.array([120, 122, 154]),
    "Emerald Green": np.array([85, 105, 138]),
    "Dark Forest Green": np.array([55, 112, 130])
}

VALIDATION_PROMPTS = [
    "a photograph of a person wearing clothing, a dress, a saree, or a fashion outfit",
    "a clean single photograph of a piece of clothing, apparel, or wearable garment",
    "a close-up portrait of only a human face without clothes visible",
    "a photo of an animal or pet",
    "a vehicle, car, bicycle, or motorcycle",
    "an electronic device, screen, laptop, or mobile phone",
    "furniture, indoor room background, or household object",
    "food, beverage, dish, or meal",
    "a document, text, paper, meme, or graphic artwork"
]

# -------------------------------------------------------------
# 3. Pipeline Core Functions with Execution Timing
# -------------------------------------------------------------
def crop_to_garment(input_path: str, output_path: str = None) -> str:
    start_t = time.perf_counter()
    if output_path is None:
        output_path = input_path

    logger.debug("[crop_to_garment] Processing: %s", input_path)
    try:
        img = Image.open(input_path).convert("RGB")
        orig_w, orig_h = img.size

        max_dim = 1024
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            logger.debug("[crop_to_garment] Downscaled from (%d, %d) to %s", orig_w, orig_h, img.size)

        nobg = remove(img, session=rembg_session)
        bbox = nobg.getbbox()

        if bbox:
            cropped = img.crop(bbox)
            cropped.save(output_path, "JPEG", quality=95)
            logger.info(
                "[crop_to_garment] Bounding box cropped: %s (took %.3fs)",
                bbox, time.perf_counter() - start_t
            )
        else:
            logger.warning("[crop_to_garment] No distinct alpha bbox found. Saving resized image.")
            img.save(output_path, "JPEG", quality=95)

    except Exception as e:
        logger.error("[crop_to_garment] Segmentation failure (%s): %s", e, traceback.format_exc())
        if input_path != output_path and os.path.exists(input_path):
            shutil.copyfile(input_path, output_path)

    return output_path

def is_garment(image_path: str, min_confidence: float = 0.45) -> Tuple[bool, float, str]:
    start_t = time.perf_counter()
    logger.debug("[is_garment] Verifying apparel authenticity: %s", image_path)

    image = Image.open(image_path).convert("RGB")
    inputs = models["clip_processor"](
        text=VALIDATION_PROMPTS, images=image, return_tensors="pt", padding=True
    ).to(models["device"])

    with torch.no_grad():
        outputs = models["clip_model"](**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

    best_idx = int(np.argmax(probs))
    confidence = float(probs[best_idx])
    combined_apparel_prob = float(probs[0] + probs[1])
    is_valid = (best_idx in [0, 1]) and (combined_apparel_prob >= min_confidence)
    detected_label = "Apparel" if is_valid else VALIDATION_PROMPTS[best_idx]

    logger.info(
        "[is_garment] Result: %s | Valid: %s | Conf: %.3f (Combined Apparel: %.3f) in %.3fs",
        detected_label, is_valid, confidence, combined_apparel_prob, time.perf_counter() - start_t
    )
    return is_valid, confidence, detected_label

def classify_garment_type(image_path: str) -> Tuple[str, float]:
    start_t = time.perf_counter()
    image = Image.open(image_path).convert("RGB")
    prompts = [f"a photograph of a {item}" for item in ALL_GARMENT_TYPES]
    inputs = models["clip_processor"](text=prompts, images=image, return_tensors="pt", padding=True).to(models["device"])

    with torch.no_grad():
        outputs = models["clip_model"](**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

    best_idx = int(np.argmax(probs))
    selected_type = ALL_GARMENT_TYPES[best_idx].title()
    prob = float(probs[best_idx])

    logger.info("[classify_garment_type] Identified: '%s' (Conf: %.3f) in %.3fs", selected_type, prob, time.perf_counter() - start_t)
    return selected_type, prob

def classify_pattern_type(image_path: str) -> Tuple[str, float]:
    start_t = time.perf_counter()
    image = Image.open(image_path).convert("RGB")
    prompts = [f"clothing fabric with a {pattern}" for pattern in ALL_PATTERNS]
    inputs = models["clip_processor"](text=prompts, images=image, return_tensors="pt", padding=True).to(models["device"])

    with torch.no_grad():
        outputs = models["clip_model"](**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

    best_idx = int(np.argmax(probs))
    selected_pattern = ALL_PATTERNS[best_idx].title()
    prob = float(probs[best_idx])

    logger.info("[classify_pattern_type] Identified: '%s' (Conf: %.3f) in %.3fs", selected_pattern, prob, time.perf_counter() - start_t)
    return selected_pattern, prob

def extract_visual_embedding(image_path: str) -> np.ndarray:
    start_t = time.perf_counter()
    image = Image.open(image_path).convert("RGB")
    inputs = models["clip_processor"](images=image, return_tensors="pt").to(models["device"])

    with torch.no_grad():
        outputs = models["clip_model"].get_image_features(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            tensor = outputs.pooler_output
        elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            tensor = outputs.image_embeds
        elif isinstance(outputs, (tuple, list)):
            tensor = outputs[0]
        else:
            tensor = outputs
        features = tensor.squeeze().cpu().numpy()

    norm = np.linalg.norm(features)
    normalized = (features / norm) if norm > 0 else features

    logger.debug("[extract_visual_embedding] Generated 512-D vector (L2 norm: %.4f) in %.3fs", norm, time.perf_counter() - start_t)
    return normalized

def analyze_dress_shade(image_path: str) -> Tuple[str, np.ndarray]:
    start_t = time.perf_counter()
    img = cv2.imread(image_path)
    if img is None:
        logger.warning("[analyze_dress_shade] Could not open image: %s. Returning Unknown.", image_path)
        return "Unknown", np.array([128, 128, 128])

    h, w, _ = img.shape
    y1, y2 = int(h * 0.20), int(h * 0.80)
    x1, x2 = int(w * 0.25), int(w * 0.75)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        crop = img

    crop = cv2.resize(crop, (150, 150))
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    mask = (hsv[:, :, 1] > 25) & (hsv[:, :, 2] > 30) & (hsv[:, :, 2] < 245)
    valid_pixels = lab[mask]

    if len(valid_pixels) < 200:
        logger.debug("[analyze_dress_shade] Low saturation mask (<200 pixels). Falling back to center crop.")
        valid_pixels = lab[35:115, 35:115].reshape(-1, 3)

    n_clusters = 3 if len(valid_pixels) >= 3 else 1
    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    kmeans.fit(valid_pixels)
    counts = np.bincount(kmeans.labels_)
    dominant_lab = kmeans.cluster_centers_[np.argmax(counts)]

    closest_name = "Custom Shade"
    min_dist = float("inf")
    for name, ref_lab in COLOR_PALETTE.items():
        dL = dominant_lab[0] - ref_lab[0]
        da = dominant_lab[1] - ref_lab[1]
        db = dominant_lab[2] - ref_lab[2]
        dist = np.sqrt((0.4 * (dL ** 2)) + (1.3 * (da ** 2)) + (1.3 * (db ** 2)))
        if dist < min_dist:
            min_dist = dist
            closest_name = name

    logger.info(
        "[analyze_dress_shade] Matched: '%s' (LAB: %s, Delta-E dist: %.2f) in %.3fs",
        closest_name, np.round(dominant_lab, 1).tolist(), min_dist, time.perf_counter() - start_t
    )
    return closest_name, dominant_lab

def compute_shade_similarity(lab1: np.ndarray, lab2: np.ndarray) -> float:
    dL = lab1[0] - lab2[0]
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]
    dist = np.sqrt((0.35 * (dL ** 2)) + (1.35 * (da ** 2)) + (1.35 * (db ** 2)))
    score = 1.0 - (dist / 85.0)
    return float(np.clip(score, 0.0, 1.0))

def get_file_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_all_inventory_orm(db: Session) -> list:
    start_t = time.perf_counter()
    items = db.query(GarmentItem).all()
    inventory = []
    for item in items:
        try:
            inventory.append({
                "id": item.id,
                "user_id": item.user_id,
                "dress_name": item.dress_name,
                "filename": item.filename,
                "garment_type": item.garment_type,
                "shade_name": item.shade_name,
                "pattern_type": item.pattern_type,
                "dominant_lab": np.array(json.loads(item.dominant_lab)),
                "pattern_emb": np.array(json.loads(item.pattern_emb), dtype=np.float32)
            })
        except Exception as e:
            logger.error("[get_all_inventory_orm] Corrupted record skipped (ID: %s): %s", item.id, e)

    logger.debug("[get_all_inventory_orm] Loaded %d inventory items in %.3fs", len(inventory), time.perf_counter() - start_t)
    return inventory

# -------------------------------------------------------------
# 4. Web Dashboard Interface
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(db: Session = Depends(get_db)):
    logger.info("[GET /] Serving Web Dashboard")
    items = get_all_inventory_orm(db)

    catalog_cards = "".join(
        f'<div style="text-align:center; display:inline-block; margin:8px; padding:8px; border:1px solid #e2e8f0; border-radius:8px; background:#fff; width:150px; vertical-align:top;">'
        f'<img src="/uploads/{item["filename"]}" style="width:100%; height:130px; object-fit:cover; border-radius:6px;"/><br>'
        f'<b style="font-size:12px; display:block; height:32px; overflow:hidden; margin-top:4px;">{item["dress_name"]}</b>'
        f'<span style="font-size:10px; background:#e0f2fe; color:#0369a1; padding:2px 4px; border-radius:4px; display:block; margin:2px 0;">{item["garment_type"]}</span>'
        f'<span style="font-size:10px; background:#fdf2f8; color:#be185d; padding:2px 4px; border-radius:4px; display:block; margin:2px 0;">{item["pattern_type"]}</span>'
        f'<span style="font-size:10px; background:#f1f5f9; padding:2px 4px; border-radius:4px; display:block; margin-bottom:6px;">{item["shade_name"]}</span>'
        f'<form action="/delete-item/{item["id"]}" method="post" onsubmit="return confirm(\'Delete this garment?\');">'
        f'<button type="submit" style="background:#ef4444; color:white; border:none; padding:4px 8px; border-radius:4px; font-size:11px; cursor:pointer; width:100%;">Delete</button>'
        f'</form>'
        f'</div>'
        for item in items
    )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Universal Apparel Matcher</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 960px; margin: 30px auto; padding: 0 20px; color: #1e293b; background: #f8fafc; }}
            .card {{ background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
            button {{ background: #0284c7; color: white; border: none; padding: 10px 18px; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 600; }}
            button:hover {{ background: #0369a1; }}
            input[type="file"], input[type="text"], select {{ margin-bottom: 12px; }}
            .grid {{ max-height: 440px; overflow-y: auto; padding: 10px; border: 1px dashed #cbd5e1; border-radius: 8px; background: #fafafa; }}
        </style>
    </head>
    <body>
        <h1>👔 Universal Apparel Matcher</h1>
        <p style="color: #64748b;">Powered by CLIP Vision & Texture Classification</p>

        <div class="card">
            <h3>➕ Add Garments to Inventory (Bulk Upload)</h3>
            <form action="/bulk-upload" method="post" enctype="multipart/form-data">
                <label><b>Select Clothes Images:</b></label><br><br>
                <input type="file" name="files" accept="image/*" multiple required><br>
                <button type="submit">Scan, Classify & Add to Stock</button>
            </form>
            <hr style="margin: 20px 0; border: none; border-top: 1px solid #e2e8f0;">
            <h4>Stored Inventory ({len(items)} Items)</h4>
            <div class="grid">
                {catalog_cards if catalog_cards else "<i>No clothes in inventory yet. Add some above.</i>"}
            </div>
        </div>

        <div class="card">
            <h3>🔍 Match Any Garment</h3>
            <form action="/compare-ui" method="post" enctype="multipart/form-data">
                <label><b>Upload Garment to Compare:</b></label><br>
                <input type="file" name="file" accept="image/*" required><br><br>
                <label>Match Sensitivity Threshold (default: 0.62): </label>
                <input type="number" step="0.05" min="0.1" max="1.0" name="threshold" value="0.62" style="width: 70px;"><br><br>
                <button type="submit" style="background: #16a34a;">Find Closest Pattern & Shade Match</button>
            </form>
        </div>
    </body>
    </html>
    """

# -------------------------------------------------------------
# 5. Bulk Upload Route (ORM)
# -------------------------------------------------------------
@app.post("/bulk-upload", response_class=HTMLResponse)
def bulk_upload(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    batch_start = time.perf_counter()
    logger.info("[POST /bulk-upload] Incoming upload batch of %d files", len(files))

    inventory = get_all_inventory_orm(db)
    existing_hashes = set()
    for item in inventory:
        path = os.path.join(UPLOAD_DIR, item["filename"])
        if os.path.exists(path):
            existing_hashes.add(get_file_sha256(path))

    accepted_count = 0
    rejected_non_garment = 0
    rejected_duplicates = []

    for idx, file in enumerate(files, start=1):
        file_start = time.perf_counter()
        logger.info("[bulk-upload] Processing item %d/%d: '%s'", idx, len(files), file.filename)

        ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
        dress_id = str(uuid.uuid4())
        saved_filename = f"{dress_id}{ext}"
        file_path = os.path.join(UPLOAD_DIR, saved_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        valid_apparel, conf, detected_as = is_garment(file_path, min_confidence=0.52)
        if not valid_apparel:
            logger.warning("[bulk-upload] File '%s' rejected by validation gate: %s", file.filename, detected_as)
            if os.path.exists(file_path):
                os.remove(file_path)
            rejected_non_garment += 1
            continue

        file_hash = get_file_sha256(file_path)
        if file_hash in existing_hashes:
            logger.warning("[bulk-upload] File '%s' rejected: Exact SHA-256 hash collision", file.filename)
            if os.path.exists(file_path):
                os.remove(file_path)
            rejected_duplicates.append(file.filename)
            continue

        crop_to_garment(file_path)

        garment_type, _ = classify_garment_type(file_path)
        pattern_type, _ = classify_pattern_type(file_path)
        shade_name, dominant_lab = analyze_dress_shade(file_path)
        pattern_emb = extract_visual_embedding(file_path)

        is_vis_dup = False
        for inv in inventory:
            if inv["garment_type"].lower() == garment_type.lower() and inv["pattern_type"].lower() == pattern_type.lower():
                cos_sim = float(np.dot(pattern_emb, inv["pattern_emb"]))
                shade_sim = compute_shade_similarity(dominant_lab, inv["dominant_lab"])
                if cos_sim >= 0.985 and shade_sim >= 0.96:
                    is_vis_dup = True
                    logger.warning(
                        "[bulk-upload] Visual duplicate: '%s' matches '%s' (Cos: %.3f, Shade: %.3f)",
                        file.filename, inv['dress_name'], cos_sim, shade_sim
                    )
                    rejected_duplicates.append(f"{file.filename} (Visual twin of '{inv['dress_name']}')")
                    break

        if is_vis_dup:
            if os.path.exists(file_path):
                os.remove(file_path)
            continue

        item_title = os.path.splitext(file.filename)[0].replace("_", " ").title()

        new_item = GarmentItem(
            id=dress_id,
            user_id="default_user",
            dress_name=item_title,
            filename=saved_filename,
            garment_type=garment_type,
            shade_name=shade_name,
            pattern_type=pattern_type,
            dominant_lab=json.dumps(dominant_lab.tolist()),
            pattern_emb=json.dumps(pattern_emb.tolist())
        )
        db.add(new_item)

        inventory.append({
            "id": dress_id, "dress_name": item_title, "filename": saved_filename,
            "garment_type": garment_type, "shade_name": shade_name, "pattern_type": pattern_type,
            "dominant_lab": dominant_lab, "pattern_emb": pattern_emb
        })
        existing_hashes.add(file_hash)
        accepted_count += 1
        logger.info("[bulk-upload] Registered '%s' as %s in %.3fs", item_title, dress_id, time.perf_counter() - file_start)

    db.commit()
    logger.info(
        "[bulk-upload] Batch complete: %d added, %d non-garment, %d duplicates in %.3fs",
        accepted_count, rejected_non_garment, len(rejected_duplicates), time.perf_counter() - batch_start
    )

    dup_list = "".join([f"<li>{name}</li>" for name in rejected_duplicates])

    return f"""
    <html>
    <body style="font-family: sans-serif; max-width: 600px; margin: 50px auto; padding: 0 20px; color: #1e293b;">
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px;">
            <h2>Upload Processing Summary</h2>
            <p style="color: #16a34a; font-size: 1.1rem;">✅ <b>{accepted_count}</b> unique garments added.</p>
            {"<p style='color: #dc2626;'>❌ <b>" + str(rejected_non_garment) + "</b> non-clothing images rejected.</p>" if rejected_non_garment else ""}
            {"<div style='background: #fffbeb; border: 1px solid #fef3c7; padding: 12px; border-radius: 8px;'><p style='color: #b45309; margin: 0 0 6px 0;'><b>⚠️ " + str(len(rejected_duplicates)) + " Duplicates Skipped:</b></p><ul style='margin:0; padding-left: 20px; font-size: 13px; color: #78350f;'>" + dup_list + "</ul></div>" if rejected_duplicates else ""}
            <br><a href="/" style="display:inline-block; padding: 10px 20px; background:#0284c7; color:white; border-radius:6px; text-decoration:none;">← Back to Dashboard</a>
        </div>
    </body>
    </html>
    """

# -------------------------------------------------------------
# 6. Delete Garment Route (ORM)
# -------------------------------------------------------------
@app.post("/delete-item/{item_id}")
async def delete_inventory_item(item_id: str, db: Session = Depends(get_db)):
    logger.info("[POST /delete-item] Request to delete ID: %s", item_id)
    item = db.query(GarmentItem).filter(GarmentItem.id == item_id).first()
    if item:
        file_path = os.path.join(UPLOAD_DIR, item.filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info("[delete-item] Deleted associated file: %s", file_path)
            except Exception as e:
                logger.error("[delete-item] Failed removing file %s: %s", file_path, e)
        db.delete(item)
        db.commit()
        logger.info("[delete-item] Successfully deleted DB record %s", item_id)
    else:
        logger.warning("[delete-item] ID %s not found in DB", item_id)
    return RedirectResponse(url="/", status_code=303)

# -------------------------------------------------------------
# 7. Native Mobile Endpoint: Add Item (ORM)
# -------------------------------------------------------------
@app.post("/api/add-garment")
async def api_add_garment(
    file: UploadFile = File(...),
    dress_name: str = Form(...),
    user_id: str = Form("default_user"),
    db: Session = Depends(get_db)
):
    start_t = time.perf_counter()
    logger.info("[POST /api/add-garment] User '%s' adding '%s' (%s)", user_id, dress_name, file.filename)

    try:
        ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
        dress_id = str(uuid.uuid4())
        saved_filename = f"{dress_id}{ext}"
        file_path = os.path.join(UPLOAD_DIR, saved_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        valid_apparel, conf, detected_as = is_garment(file_path, min_confidence=0.45)
        if not valid_apparel:
            logger.warning("[/api/add-garment] Rejected non-garment (%s, conf: %.2f)", detected_as, conf)
            if os.path.exists(file_path):
                os.remove(file_path)
            return JSONResponse({
                "status": "rejected",
                "message": f"Rejected: Identified as {detected_as}"
            }, status_code=422)

        crop_to_garment(file_path)

        garment_type, _ = classify_garment_type(file_path)
        pattern_type, _ = classify_pattern_type(file_path)
        shade_name, dominant_lab = analyze_dress_shade(file_path)
        pattern_emb = extract_visual_embedding(file_path)

        new_garment = GarmentItem(
            id=dress_id,
            user_id=user_id,
            dress_name=dress_name.strip(),
            filename=saved_filename,
            garment_type=garment_type,
            shade_name=shade_name,
            pattern_type=pattern_type,
            dominant_lab=json.dumps(dominant_lab.tolist()),
            pattern_emb=json.dumps(pattern_emb.tolist())
        )
        db.add(new_garment)
        db.commit()
        db.refresh(new_garment)

        logger.info(
            "[/api/add-garment] Successfully stored '%s' (%s) in %.3fs",
            dress_name, dress_id, time.perf_counter() - start_t
        )

        return {
            "status": "success",
            "message": "Garment successfully registered",
            "item": {
                "id": new_garment.id,
                "user_id": new_garment.user_id,
                "name": new_garment.dress_name,
                "garment_type": new_garment.garment_type,
                "shade_name": new_garment.shade_name,
                "pattern_type": new_garment.pattern_type,
                "image_url": f"/uploads/{saved_filename}"
            }
        }
    except Exception as e:
        db.rollback()
        logger.error("[/api/add-garment] Error: %s\n%s", e, traceback.format_exc())
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# -------------------------------------------------------------
# 8. Native Mobile Endpoint: Match Garment (ORM)
# -------------------------------------------------------------
@app.post("/api/match-garment")
async def api_match_garment(
    file: UploadFile = File(...),
    threshold: float = Form(0.50),
    user_id: str = Form(None),
    db: Session = Depends(get_db)
):
    start_t = time.perf_counter()
    logger.info("[POST /api/match-garment] Query from user: %s (Threshold: %.2f)", user_id, threshold)

    try:
        raw_inventory = get_all_inventory_orm(db)
        if user_id:
            raw_inventory = [i for i in raw_inventory if i.get("user_id") == user_id]

        if not raw_inventory:
            logger.warning("[/api/match-garment] Inventory pool is empty.")
            return JSONResponse({"status": "error", "message": "Inventory is empty"}, status_code=400)

        seen = set()
        inventory = [item for item in raw_inventory if not (item["filename"] in seen or seen.add(item["filename"]))]

        query_filename = f"query_{uuid.uuid4().hex}.jpg"
        query_path = os.path.join(UPLOAD_DIR, query_filename)

        image = Image.open(file.file).convert("RGB")
        image.save(query_path, "JPEG")
        crop_to_garment(query_path)

        valid_apparel, conf, detected_as = is_garment(query_path, min_confidence=0.45)
        if not valid_apparel:
            logger.warning("[/api/match-garment] Query image rejected: Non-clothing (%s)", detected_as)
            if os.path.exists(query_path):
                os.remove(query_path)
            return JSONResponse({
                "status": "rejected",
                "message": f"Non-clothing item detected: {detected_as}"
            }, status_code=422)

        query_type, _ = classify_garment_type(query_path)
        query_shade, query_lab = analyze_dress_shade(query_path)
        query_pattern, _ = classify_pattern_type(query_path)
        query_emb = extract_visual_embedding(query_path)

        primary_matches = []
        secondary_matches = []

        q_type_lower = query_type.lower()
        is_query_button_shirt = any(k in q_type_lower for k in ["casual shirt", "formal dress shirt"])

        for item in inventory:
            i_type = item["garment_type"].lower()
            name_lower = item["dress_name"].lower()

            is_outfit_set = any(k in name_lower for k in ["trench coat", "set", "outfit", "suit", "jacket"]) and not ("shirt" in name_lower)
            is_same_shirt = is_query_button_shirt and ("casual shirt" in i_type or "formal dress shirt" in i_type)
            is_exact_type = (i_type == q_type_lower) or is_same_shirt

            shade_sim = compute_shade_similarity(query_lab, item["dominant_lab"])
            pattern_vis_sim = max(0.0, float(np.dot(query_emb, item["pattern_emb"])))
            exact_pattern = (item["pattern_type"].lower() == query_pattern.lower())

            color_factor = max(0.0, (shade_sim - 0.20) / 0.80)
            cat_weight = 0.25 if is_exact_type else 0.0
            pattern_weight = 0.25 if exact_pattern else 0.0

            total_score = (0.40 * shade_sim) + (0.15 * pattern_vis_sim) + (color_factor * (cat_weight + pattern_weight))
            match_pct = round(min(total_score, 1.0) * 100, 1)
            is_strong = (total_score >= threshold) and (shade_sim >= 0.50)

            result_entry = {
                "id": item["id"],
                "name": item["dress_name"],
                "image_url": f"/uploads/{item['filename']}",
                "garment_type": item["garment_type"],
                "shade_name": item["shade_name"],
                "shade_similarity": round(shade_sim * 100, 1),
                "pattern_type": item["pattern_type"],
                "pattern_similarity": round(pattern_vis_sim * 100, 1),
                "match_percentage": match_pct,
                "badge": "STRONG" if is_strong else "WEAK"
            }

            if is_exact_type and not is_outfit_set:
                primary_matches.append(result_entry)
            else:
                secondary_matches.append(result_entry)

        primary_matches.sort(key=lambda x: x["match_percentage"], reverse=True)
        secondary_matches.sort(key=lambda x: x["match_percentage"], reverse=True)

        logger.info(
            "[/api/match-garment] Returning %d primary, %d secondary matches in %.3fs",
            len(primary_matches), len(secondary_matches), time.perf_counter() - start_t
        )

        return {
            "status": "success",
            "target": {
                "garment_type": query_type,
                "shade": query_shade,
                "pattern": query_pattern,
                "image_url": f"/uploads/{query_filename}"
            },
            "primary_matches": primary_matches,
            "secondary_matches": secondary_matches
        }

    except Exception as e:
        logger.error("[/api/match-garment] Matching pipeline error: %s\n%s", e, traceback.format_exc())
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# -------------------------------------------------------------
# 9. Web Comparison View (ORM)
# -------------------------------------------------------------
@app.post("/compare-ui", response_class=HTMLResponse)
def compare_dress_ui(file: UploadFile = File(...), threshold: float = Form(0.50), db: Session = Depends(get_db)):
    start_t = time.perf_counter()
    logger.info("[POST /compare-ui] Web comparison triggered")

    try:
        raw_inventory = get_all_inventory_orm(db)
        if not raw_inventory:
            logger.warning("[compare-ui] Comparison blocked: Empty inventory")
            return "<h3>Inventory is empty. Add clothes first!</h3><a href='/'>Go Back</a>"

        seen = set()
        inventory = [item for item in raw_inventory if not (item["filename"] in seen or seen.add(item["filename"]))]

        query_filename = f"query_{uuid.uuid4().hex}.jpg"
        query_path = os.path.join(UPLOAD_DIR, query_filename)

        image = Image.open(file.file).convert("RGB")
        image.save(query_path, "JPEG")
        crop_to_garment(query_path)

        valid_apparel, conf, detected_as = is_garment(query_path, min_confidence=0.52)
        if not valid_apparel:
            logger.warning("[compare-ui] Validation rejected: %s (conf: %.2f)", detected_as, conf)
            if os.path.exists(query_path):
                os.remove(query_path)
            return f"""
            <div style="font-family: sans-serif; max-width: 500px; margin: 50px auto; text-align: center; padding: 24px; background: #fef2f2; border: 1px solid #fecaca; border-radius: 12px;">
                <h2 style="color: #dc2626;">❌ Garment Not Detected</h2>
                <p>Detected as: <b>{detected_as.title()}</b></p>
                <a href="/" style="display:inline-block; margin-top:15px; padding:10px 20px; background:#1e293b; color:white; border-radius:6px; text-decoration:none;">← Try Another</a>
            </div>
            """

        query_type, _ = classify_garment_type(query_path)
        query_shade, query_lab = analyze_dress_shade(query_path)
        query_pattern, _ = classify_pattern_type(query_path)
        query_emb = extract_visual_embedding(query_path)

        primary_candidates = []
        secondary_candidates = []

        q_type_lower = query_type.lower()
        is_query_button_shirt = any(k in q_type_lower for k in ["casual shirt", "formal dress shirt"])

        for item in inventory:
            i_type = item["garment_type"].lower()
            name_lower = item["dress_name"].lower()

            is_outfit_set = any(k in name_lower for k in ["trench coat", "set", "outfit", "suit", "jacket"]) and not ("shirt" in name_lower)
            is_same_shirt = is_query_button_shirt and ("casual shirt" in i_type or "formal dress shirt" in i_type)
            is_exact_type = (i_type == q_type_lower) or is_same_shirt

            shade_sim = compute_shade_similarity(query_lab, item["dominant_lab"])
            pattern_vis_sim = max(0.0, float(np.dot(query_emb, item["pattern_emb"])))
            exact_pattern = (item["pattern_type"].lower() == query_pattern.lower())

            color_factor = max(0.0, (shade_sim - 0.20) / 0.80)
            cat_weight = 0.25 if is_exact_type else 0.0
            pattern_weight = 0.25 if exact_pattern else 0.0

            total_score = (0.40 * shade_sim) + (0.15 * pattern_vis_sim) + (color_factor * (cat_weight + pattern_weight))
            match_pct = round(min(total_score, 1.0) * 100, 1)
            is_strong = (total_score >= threshold) and (shade_sim >= 0.50)

            res_data = {
                "item": item,
                "total_score": total_score,
                "match_pct": match_pct,
                "shade_sim": round(shade_sim * 100, 1),
                "pattern_sim": round(pattern_vis_sim * 100, 1),
                "is_match": is_strong
            }

            if is_exact_type and not is_outfit_set:
                primary_candidates.append(res_data)
            else:
                secondary_candidates.append(res_data)

        primary_candidates.sort(key=lambda x: x["total_score"], reverse=True)
        secondary_candidates.sort(key=lambda x: x["total_score"], reverse=True)

        logger.info(
            "[compare-ui] Match rendered in %.3fs (%d primary, %d secondary)",
            time.perf_counter() - start_t, len(primary_candidates), len(secondary_candidates)
        )

        def build_cards(candidates_list, limit=8):
            if not candidates_list:
                return "<p style='color: #64748b; font-style: italic; padding: 10px;'>No items found in this section.</p>"
            html = ""
            for rank, res in enumerate(candidates_list[:limit], start=1):
                item = res["item"]
                badge = '<span style="background:#dcfce7; color:#15803d; padding:2px 8px; border-radius:4px; font-weight:bold; font-size:12px;">STRONG</span>' if res["is_match"] else '<span style="background:#fee2e2; color:#b91c1c; padding:2px 8px; border-radius:4px; font-weight:bold; font-size:12px;">WEAK</span>'
                html += f"""
                <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; width: 220px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 11px; font-weight: bold; color: #64748b;">#{rank} Ranked</span>
                        {badge}
                    </div>
                    <img src="/uploads/{item['filename']}" style="width: 100%; height: 200px; object-fit: cover; border-radius: 6px; border: 1px solid #f1f5f9;">
                    <h4 style="margin: 8px 0 4px 0; font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{item['dress_name']}</h4>
                    <p style="font-size: 18px; font-weight: bold; color: #0284c7; margin: 4px 0;">{res['match_pct']}%</p>
                    <div style="background: #f8fafc; border-radius: 6px; padding: 8px; font-size: 11px; text-align: left; margin-top: 8px; color: #334155;">
                        <div><b>Type:</b> {item['garment_type']}</div>
                        <div><b>Shade:</b> {item['shade_name']} ({res['shade_sim']}%)</div>
                        <div><b>Pattern:</b> {item['pattern_type']} ({res['pattern_sim']}%)</div>
                    </div>
                </div>
                """
            return html

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Match Results</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 1150px; margin: 30px auto; padding: 0 20px; color: #1e293b; background: #f8fafc; }}
                .container {{ display: flex; gap: 24px; align-items: flex-start; }}
                .query-panel {{ flex: 0 0 280px; background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; text-align: center; position: sticky; top: 20px; }}
                .results-panel {{ flex: 1; }}
                .grid {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 30px; }}
                .section-title {{ font-size: 18px; font-weight: bold; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 16px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="query-panel">
                    <h3 style="margin-top:0;">Target Garment</h3>
                    <img src="/uploads/{query_filename}" style="width: 230px; height: 270px; object-fit: cover; border-radius: 8px; border: 1px solid #ddd;"><br>
                    <div style="text-align: left; background: #f8fafc; border-radius: 8px; padding: 12px; margin-top: 14px; font-size: 13px;">
                        <p style="margin: 4px 0;"><b>Type:</b> <span style="color: #4f46e5;">{query_type}</span></p>
                        <p style="margin: 4px 0;"><b>Shade:</b> <span style="color: #0284c7;">{query_shade}</span></p>
                        <p style="margin: 4px 0;"><b>Pattern:</b> <span style="color: #be185d;">{query_pattern}</span></p>
                    </div>
                    <a href="/" style="display: inline-block; margin-top: 18px; padding: 8px 16px; background: #1e293b; color: white; border-radius: 6px; text-decoration: none; font-size: 13px;">← Search New Item</a>
                </div>

                <div class="results-panel">
                    <div class="section-title" style="color: #0f172a;">
                        🥇 Primary Matches ({query_type})
                    </div>
                    <div class="grid">
                        {build_cards(primary_candidates, limit=8)}
                    </div>

                    <div class="section-title" style="color: #475569; margin-top: 30px;">
                        🥈 Secondary Alternatives (Sets, Outfits & Other Types)
                    </div>
                    <div class="grid">
                        {build_cards(secondary_candidates, limit=8)}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
    except Exception:
        err = traceback.format_exc()
        logger.error("[compare-ui] Unhandled Exception: %s", err)
        return f"<pre>{err}</pre>"