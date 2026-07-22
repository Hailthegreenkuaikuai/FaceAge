import os
import sys
import json
import uuid
import threading
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "test"))

import keras
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

from predict_folder_demo import (
    get_face_bbox_from_image,
    preprocess_face,
    get_model_prediction_batch,
)

# --- paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "stored_faces")
PHOTOS_JSON = os.path.join(BASE_DIR, "outputs", "photos.json")
RESULTS_JSON = os.path.join(BASE_DIR, "outputs", "results.json")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- model ---
MODEL_PATH = os.path.join(BASE_DIR, "models", "faceage_model.keras")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "models", "faceage_model.h5")

print(f"Loading model from {MODEL_PATH}...")
model = keras.models.load_model(MODEL_PATH, safe_mode=False)
print("Model loaded.")

# --- concurrency ---
predict_lock = threading.Lock()   # serialize MTCNN + Keras inference
photos_lock = threading.Lock()
results_lock = threading.Lock()

# --- app ---
app = FastAPI()


def append_json_line(filepath: str, record: dict, lock: threading.Lock):
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with lock:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(line)


@app.post("/face/age")
async def face_age(files: list[UploadFile] = File(...)):
    now = datetime.now(timezone.utc).isoformat()
    result_id = uuid.uuid4().hex
    photo_ids: list[str] = []
    face_idx_map: list[int] = []  # file indices that have a face

    # --- step 1: read and save photos (async, outside lock) ---
    saved: list[tuple[int, str, str]] = []  # (index, filepath, original_name)

    for i, file in enumerate(files):
        photo_id = uuid.uuid4().hex[:20]
        photo_ids.append(photo_id)

        ext = os.path.splitext(file.filename or "photo.jpg")[1] or ".jpg"
        renamed = f"{photo_id}{ext}"
        filepath = os.path.join(UPLOAD_DIR, renamed)

        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)

        saved.append((i, filepath, file.filename or "photo.jpg"))

    # --- step 2: detect faces + predict (serialized, no await inside lock) ---
    with predict_lock:
        faces_batch = []

        for idx, filepath, original_name in saved:
            bbox = get_face_bbox_from_image(filepath)
            if not bbox:
                continue

            face_idx_map.append(idx)
            faces_batch.append(preprocess_face(filepath, bbox))

        if faces_batch:
            predictions = np.atleast_1d(
                get_model_prediction_batch(model, faces_batch)
            )

    # --- step 3: build results and write records ---
    pred_lookup = dict(zip(face_idx_map, predictions)) if faces_batch else {}

    individual_ages: list[dict] = []
    valid_ages: list[float] = []

    for idx, filepath, original_name in saved:
        ext = os.path.splitext(original_name)[1] or ".jpg"
        renamed = f"{photo_ids[idx]}{ext}"

        if idx in pred_lookup:
            age = float(pred_lookup[idx])
            status = "completed"
            valid_ages.append(age)
        else:
            age = None
            status = "no_face_detected"

        individual_ages.append({
            "original_name": original_name,
            "predicted_age": age,
            "status": status,
        })

        append_json_line(PHOTOS_JSON, {
            "photo_id": photo_ids[idx],
            "user_id": None,
            "original_name": original_name,
            "renamed_name": renamed,
            "predicted_age": age,
            "status": status,
            "created_at": now,
            "updated_at": now,
        }, photos_lock)

    # --- step 4: write result records (one per photo) ---
    for photo_id in photo_ids:
        append_json_line(RESULTS_JSON, {
            "id": uuid.uuid4().hex[:12],
            "result_id": result_id,
            "photo_id": photo_id,
            "created_at": now,
        }, results_lock)

    avg_age = sum(valid_ages) / len(valid_ages) if valid_ages else None

    return {
        "result_id": result_id,
        "avg_age": avg_age,
        "individual_ages": individual_ages,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/concurrency", response_class=HTMLResponse)
async def concurrency_test():
    with open(os.path.join(STATIC_DIR, "concurrency.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
