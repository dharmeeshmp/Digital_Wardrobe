import os
import sys
import uuid
import io
import time
import shutil
import logging
import traceback
import base64
from typing import List, Dict, Any

import numpy as np
from PIL import Image
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.concurrency import run_in_threadpool
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

# In-memory status store for background tasks
task_store: Dict[str, Dict[str, Any]] = {}

# -------------------------------------------------------------
# Schemas
# -------------------------------------------------------------
class CompareRequest(BaseModel):
    query_embedding: List[float]
    wardrobe_embeddings: List[List[float]]
    wardrobe_ids: List[int]

class BatchUploadResponse(BaseModel):
    task_ids: List[str]

# -------------------------------------------------------------
# Background Worker Functions
# -------------------------------------------------------------
def process_single_garment_task(task_id: str, image_bytes: bytes, filename: str):
    """Processes heavy AI segmentation and feature extraction in a background thread."""
    task_store[task_id]["status"] = "processing"
    temp_path = f"temp_bg_{task_id}.png"

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))

        # 1. Background removal / clothing parsing (auto-handles flat-lay vs human pose)
        cleaned_img = vision_service.process_any_garment(pil_img)

        # 2. Persist processed cutout into server storage
        output_dir = "uploads/processed"
        os.makedirs(output_dir, exist_ok=True)
        safe_name = os.path.splitext(filename)[0].replace(" ", "_")
        out_filename = f"{uuid.uuid4().hex[:8]}_{safe_name}.png"
        out_path = os.path.join(output_dir, out_filename)
        cleaned_img.save(out_path, "PNG", optimize=True)

        # Save temporary file for analysis functions requiring a file path
        cleaned_img.save(temp_path, "PNG")

        # 3. Extract features & embeddings
        garment_type, type_conf = vision_service.classify_garment_type(temp_path)
        pattern_type, pattern_conf = vision_service.classify_pattern_type(temp_path)
        shade_name, dominant_lab = vision_service.analyze_dress_shade(temp_path)
        pattern_emb = vision_service.extract_visual_embedding(temp_path)

        # Base64 string if frontend prefers direct database storage
        buffered = io.BytesIO()
        cleaned_img.save(buffered, format="PNG")
        clean_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        task_store[task_id].update({
            "status": "completed",
            "result": {
                "image_path": out_path,
                "garment_type": garment_type,
                "type_confidence": round(type_conf, 3),
                "shade_name": shade_name,
                "dominant_lab": dominant_lab.tolist(),
                "pattern_type": pattern_type,
                "pattern_confidence": round(pattern_conf, 3),
                "pattern_emb": pattern_emb.tolist(),
                "cleaned_image_base64": clean_base64
            }
        })
        logger.info("Task %s completed successfully for %s", task_id, filename)

    except Exception as e:
        logger.error("Task %s failed: %s\n%s", task_id, e, traceback.format_exc())
        task_store[task_id].update({
            "status": "failed",
            "error": str(e)
        })
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# -------------------------------------------------------------
# Endpoints (Controllers)
# -------------------------------------------------------------
@app.get("/")
def home():
    return HTMLResponse("<h2>Apparel Controller Service is Running</h2><p>Visit <a href='/docs'>/docs</a> to inspect endpoints.</p>")

@app.post("/upload-bulk", response_model=BatchUploadResponse)
async def upload_bulk_garments(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Instantly receives batch files and runs pipeline asynchronously in the background."""
    created_task_ids = []

    for file in files:
        task_id = str(uuid.uuid4())
        image_bytes = await file.read()

        task_store[task_id] = {
            "status": "queued",
            "filename": file.filename,
            "result": None,
            "error": None
        }

        background_tasks.add_task(process_single_garment_task, task_id, image_bytes, file.filename)
        created_task_ids.append(task_id)

    return {"task_ids": created_task_ids}

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Polls the state and result of a background processing task."""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_store[task_id]

@app.post("/api/clean-dress")
async def clean_dress(file: UploadFile = File(...)):
    """Removes background and extracts garment from flat-lays or human models."""
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))

        cleaned_pil = vision_service.process_any_garment(pil_img)

        img_byte_arr = io.BytesIO()
        cleaned_pil.save(img_byte_arr, format="PNG")
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
    except Exception as e:
        logger.error("Clean-dress error: %s\n%s", e, traceback.format_exc())
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/rotate-garment")
async def rotate_garment(image_path: str = Form(...), turns: int = Form(...)):
    """Rotates the garment image file in 90-degree clockwise increments."""
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found")

    with Image.open(image_path) as img:
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

def _process_garment_sync(contents: bytes):
    """Run CPU-heavy garment processing in a worker thread."""
    start_t = time.time()
    temp_path = f"temp_{uuid.uuid4().hex}.png"
    try:
        pil_img = Image.open(io.BytesIO(contents))

        # Cap the largest dimension before rembg/OpenCV processing.
        max_dim = 1280
        largest_dimension = max(pil_img.size)
        if largest_dimension > max_dim:
            scale = max_dim / float(largest_dimension)
            new_size = (
                int(pil_img.size[0] * scale),
                int(pil_img.size[1] * scale),
            )
            pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
            logger.info("Resized input image to %s", new_size)

        t_seg = time.time()
        cleaned_img = vision_service.process_any_garment(pil_img)
        if cleaned_img is None or not hasattr(cleaned_img, "save"):
            logger.warning("Pipeline returned invalid image; using original upload as fallback.")
            cleaned_img = pil_img.convert("RGBA")
        if cleaned_img.mode not in ("RGB", "RGBA"):
            cleaned_img = cleaned_img.convert("RGBA")
        cleaned_img.save(temp_path, "PNG", compress_level=3)
        seg_time = time.time() - t_seg

        # Existing feature extractors operate on the persisted processed image.
        t_feat = time.time()
        garment_type, type_conf = vision_service.classify_garment_type(temp_path)
        pattern_type, pattern_conf = vision_service.classify_pattern_type(temp_path)
        shade_name, dominant_lab = vision_service.analyze_dress_shade(temp_path)
        pattern_emb = vision_service.extract_visual_embedding(temp_path)
        feat_time = time.time() - t_feat

        buffered = io.BytesIO()
        cleaned_img.save(buffered, format="PNG", compress_level=3)
        clean_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        if hasattr(pattern_emb, "tolist"):
            pattern_emb = pattern_emb.tolist()

        logger.info(
            "Pipeline finished: Seg=%.2fs, Feat=%.2fs, Total=%.2fs",
            seg_time,
            feat_time,
            time.time() - start_t,
        )

        return {
            "status": "success",
            "garment_type": garment_type,
            "type_confidence": round(type_conf, 3),
            "shade_name": shade_name,
            "dominant_lab": dominant_lab.tolist(),
            "pattern_type": pattern_type,
            "pattern_confidence": round(pattern_conf, 3),
            "pattern_emb": pattern_emb,
            "cleaned_image_base64": clean_base64
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/extract-features")
async def extract_features(
    file: UploadFile = File(...),
    clean_first: bool = Form(True)
):
    """Read the upload asynchronously and offload CPU-heavy processing."""
    try:
        contents = await file.read()
        return await run_in_threadpool(_process_garment_sync, contents)
    except Exception as e:
        logger.error("Extraction error: %s\n%s", e, traceback.format_exc())
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/bulk-upload")
async def bulk_upload(files: List[UploadFile] = File(...)):
    """Synchronous legacy bulk upload."""
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