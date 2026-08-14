import os, io
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from analysis import run_comparative_analysis
from model_utils import (
    load_model,
    preprocess_stack_for_prediction,
    preprocess_stack_for_cloud,
    run_prediction,
    run_cloud_segmentation,
    make_highlighted_changes,
    make_cloud_overlay,
    ndarray_to_base64,
    classify_urban_growth,
    TARGET_H, TARGET_W,
)

app = FastAPI(title="Urban Development Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.h5")


def tif_bytes_to_rgb(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)


# ── health check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── main analysis endpoint ────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(files: list[UploadFile] = File(...)):
    if len(files) < 2:
        return JSONResponse(
            status_code=400,
            content={"error": "Please upload at least 2 .tif images."},
        )

    # 1. Read all uploaded .tif files → RGB arrays
    images_rgb: list[np.ndarray] = []
    for f in files:
        raw = await f.read()
        images_rgb.append(tif_bytes_to_rgb(raw))

    # ── Phase-1: Comparative Analysis ────────────────────────────────────────
    phase1 = run_comparative_analysis(images_rgb)

    # ── Phase-2 & 3: Model Inference ─────────────────────────────────────────
    model = load_model(MODEL_PATH)

    pred_input   = preprocess_stack_for_prediction(images_rgb)   # (1,H,W,3*N)
    cloud_input  = preprocess_stack_for_cloud(images_rgb)        # (N,H,W,3)

    pred_mask    = run_prediction(model, pred_input)             # (H,W,1)
    cloud_mask   = run_cloud_segmentation(model, cloud_input)    # (H,W) uint8

    last_rgb_orig = images_rgb[-1]
    highlighted   = make_highlighted_changes(last_rgb_orig, pred_mask)
    cloud_overlay = make_cloud_overlay(last_rgb_orig, cloud_mask)

    # Resize pred_mask to (H,W) for display
    import cv2
    pred_display = (cv2.resize(pred_mask[..., 0],
                               (last_rgb_orig.shape[1], last_rgb_orig.shape[0])) * 255
                    ).astype(np.uint8)

    # ── Phase-3: Classification ───────────────────────────────────────────────
    baseline_area   = phase1["baseline_area_acres"]
    final_area      = phase1["final_area_acres"]
    growth_pct      = round((final_area - baseline_area) / max(baseline_area, 1e-6) * 100, 2)
    new_buildings   = phase1["total_new_buildings"]
    urban_cls, bldg_cls = classify_urban_growth(growth_pct, new_buildings)

    # predicted area proxy (fraction of mask that is white)
    pred_area_acres = round(float(pred_mask.mean()) * final_area * 0.15, 4)
    pred_buildings  = max(int(pred_mask.mean() * 50), 1)

    return {
        # ── Phase-1 ──
        "phase1": {
            "monthly_records":         phase1["monthly_records"],
            "seasonal_avg":            phase1["seasonal_avg"],
            "total_new_buildings":     new_buildings,
            "total_area_change_acres": phase1["total_area_change_acres"],
            "baseline_area_acres":     baseline_area,
            "final_area_acres":        final_area,
            "baseline_building_count": phase1["baseline_building_count"],
            "final_building_count":    phase1["final_building_count"],
        },
        # ── Phase-2 ──
        "phase2": {
            "predicted_mask_b64":   ndarray_to_base64(pred_display, "L"),
            "highlighted_img_b64":  ndarray_to_base64(highlighted,  "RGB"),
            "last_input_img_b64":   ndarray_to_base64(last_rgb_orig, "RGB"),
        },
        # ── Phase-3 ──
        "phase3": {
            "cloud_mask_b64":       ndarray_to_base64(cloud_mask,    "L"),
            "cloud_overlay_b64":    ndarray_to_base64(cloud_overlay, "RGB"),
            "urban_class":          urban_cls,
            "building_class":       bldg_cls,
            "growth_percentage":    growth_pct,
            "baseline_area":        baseline_area,
            "final_area":           final_area,
            "total_new_buildings":  new_buildings,
            "pred_area_acres":      pred_area_acres,
            "pred_buildings":       pred_buildings,
        },
    }