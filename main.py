import os
import sys
import uuid
import io
import time
import shutil
import logging
import traceback
from typing import List
from pydantic import BaseModel
import numpy as np
from PIL import Image
import base64

from fastapi import Form
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse, Response
import uvicorn

from vision_service import vision_service

# -------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("ApparelMatcher.Controller")

app = FastAPI(title="Apparel API Controller")

# -------------------------------------------------------------
# Schemas
# -------------------------------------------------------------
class CompareRequest(BaseModel):
    query_embedding: List[float]
    wardrobe_embeddings: List[List[float]]
    wardrobe_ids: List[int]

# -------------------------------------------------------------
# Endpoints (Controllers)
# -------------------------------------------------------------
@app.get("/")
def home():
    return HTMLResponse("<h2>Apparel Controller Service is Running</h2><p>Visit <a href='/docs'>/docs</a> to inspect endpoints.</p>")

@app.post("/api/clean-dress")
async def clean_dress(file: UploadFile = File(...)):
    """Removes background, strips human skin, and smoothes wrinkles."""
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))

        cleaned_pil = vision_service.dewrinkle_and_cutout(pil_img)

        img_byte_arr = io.BytesIO()
        cleaned_pil.save(img_byte_arr, format="PNG")
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
    except Exception as e:
        logger.error("Clean-dress error: %s\n%s", e, traceback.format_exc())
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/rotate-garment")
async def rotate_garment(image_path: str = Form(...), turns: int = Form(...)):
    """Rotates the garment image file by 90-degree steps (1 = 90, 2 = 180, 3 = 270 clockwise)."""
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found")

    with Image.open(image_path) as img:
        # PIL rotates counter-clockwise for positive values, so turns * 90 clockwise = -turns * 90
        # or Image.Transpose methods:
        transpose_map = {
            1: Image.Transpose.ROTATE_270,  # 90 deg clockwise
            2: Image.Transpose.ROTATE_180,  # 180 deg
            3: Image.Transpose.ROTATE_90,   # 270 deg clockwise
        }
        
        op = transpose_map.get(turns % 4)
        if op:
            rotated = img.transpose(op)
            rotated.save(image_path, "PNG", optimize=True)

    return {"status": "success", "message": "Image rotated and saved"}

@app.post("/api/extract-features")
async def extract_features(
    file: UploadFile = File(...),
    clean_first: bool = Form(True)  # Default to True so all uploads get cleaned
):
    start_t = time.perf_counter()
    temp_path = f"temp_{uuid.uuid4().hex}.png"

    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))

        # 1. Strip background, isolate cloth from skin, and smooth wrinkles
        cleaned_img = vision_service.dewrinkle_and_cutout(pil_img)
        cleaned_img.save(temp_path, "PNG")

        # 2. Extract features from the cleaned garment
        garment_type, type_conf = vision_service.classify_garment_type(temp_path)
        pattern_type, pattern_conf = vision_service.classify_pattern_type(temp_path)
        shade_name, dominant_lab = vision_service.analyze_dress_shade(temp_path)
        pattern_emb = vision_service.extract_visual_embedding(temp_path)

        # 3. Encode cleaned PNG to base64 so Flutter can save it directly
        buffered = io.BytesIO()
        cleaned_img.save(buffered, format="PNG")
        clean_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "garment_type": garment_type,
            "type_confidence": round(type_conf, 3),
            "shade_name": shade_name,
            "dominant_lab": dominant_lab.tolist(),
            "pattern_type": pattern_type,
            "pattern_confidence": round(pattern_conf, 3),
            "pattern_emb": pattern_emb.tolist(),
            "cleaned_image_base64": clean_base64  # <--- Returned to Flutter
        }
    except Exception as e:
        logger.error("Extraction error: %s\n%s", e, traceback.format_exc())
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/bulk-upload")
async def bulk_upload(files: List[UploadFile] = File(...)):
    processed_items = []
    for f in files:
        temp_path = f"temp_bulk_{uuid.uuid4().hex}.jpg"
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)

            vision_service.crop_to_garment(temp_path)
            garment_type, _ = vision_service.classify_garment_type(temp_path)
            shade_name, _ = vision_service.analyze_dress_shade(temp_path)
            emb = vision_service.extract_visual_embedding(temp_path)

            processed_items.append({
                "filename": f.filename,
                "garment_type": garment_type,
                "shade_name": shade_name,
                "pattern_emb": emb.tolist()
            })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return {"status": "success", "count": len(processed_items), "items": processed_items}

@app.post("/api/compare-wardrobe")
async def compare_wardrobe(payload: CompareRequest):
    q_emb = np.array(payload.query_embedding, dtype=np.float32)
    scores = []

    for item_id, emb_list in zip(payload.wardrobe_ids, payload.wardrobe_embeddings):
        w_emb = np.array(emb_list, dtype=np.float32)
        sim = float(np.dot(q_emb, w_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(w_emb) + 1e-7))
        scores.append({"id": item_id, "similarity": round(sim, 4)})

    scores.sort(key=lambda x: x["similarity"], reverse=True)
    return {"status": "success", "matches": scores}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)