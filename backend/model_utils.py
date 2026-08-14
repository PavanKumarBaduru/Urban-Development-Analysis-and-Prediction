import numpy as np
import tensorflow as tf
from PIL import Image
import io, base64, cv2

# ── model singleton ───────────────────────────────────────────────────────────

_model = None
TARGET_H, TARGET_W = 256, 256   # resize all images to this before inference


def load_model(path: str = "models/model.h5"):
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(path, compile=False)
        print(f"[model_utils] Loaded model from {path}")
    return _model


# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess_stack_for_prediction(images_rgb: list[np.ndarray]) -> np.ndarray:
    """
    Phase-2 input shape: (1, H, W, 3*N)
    Resize every image to TARGET_H x TARGET_W, normalise to [0,1],
    then stack along the channel axis.
    """
    resized = [
        np.array(Image.fromarray(img).resize((TARGET_W, TARGET_H))) / 255.0
        for img in images_rgb
    ]
    stacked = np.concatenate(resized, axis=-1)          # (H, W, 3*N)
    return stacked[np.newaxis, ...]                      # (1, H, W, 3*N)


def preprocess_stack_for_cloud(images_rgb: list[np.ndarray]) -> np.ndarray:
    """
    Phase-3 U-Net input shape: (N, H, W, 3)
    """
    resized = [
        np.array(Image.fromarray(img).resize((TARGET_W, TARGET_H))) / 255.0
        for img in images_rgb
    ]
    return np.array(resized)                             # (N, H, W, 3)


# ── inference ─────────────────────────────────────────────────────────────────

def run_prediction(model, input_tensor: np.ndarray) -> np.ndarray:
    """Returns (H, W, 1) float32 predicted mask, values in [0,1]."""
    pred = model.predict(input_tensor, verbose=0)
    # handle possible (1, H, W, 1) output
    if pred.ndim == 4:
        pred = pred[0]
    return pred.astype(np.float32)


def run_cloud_segmentation(model, input_tensor: np.ndarray) -> np.ndarray:
    """
    Run U-Net on the stacked input.
    We pass the mean image as a proxy single-image input if the model
    was trained on single images; adjust here if your model differs.
    Returns (H, W) binary uint8 cloud mask.
    """
    # average all frames → single (1, H, W, 3) representative image
    mean_img = input_tensor.mean(axis=0, keepdims=True)   # (1, H, W, 3)
    cloud_pred = model.predict(mean_img, verbose=0)
    if cloud_pred.ndim == 4:
        cloud_pred = cloud_pred[0, ..., 0]
    elif cloud_pred.ndim == 3:
        cloud_pred = cloud_pred[..., 0]
    binary = (cloud_pred > 0.5).astype(np.uint8) * 255
    return binary                                          # (H, W) uint8


# ── post-processing & visualisation ──────────────────────────────────────────

def make_highlighted_changes(last_rgb: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """
    Overlay predicted new-construction regions (red dots) on the last input image.
    pred_mask : (H, W, 1) float32
    """
    h, w = last_rgb.shape[:2]
    mask_resized = cv2.resize(pred_mask[..., 0], (w, h))
    binary = (mask_resized > 0.5).astype(np.uint8)

    overlay = last_rgb.copy()
    overlay[binary == 1] = [255, 50, 50]   # red highlight
    blended = cv2.addWeighted(last_rgb, 0.6, overlay, 0.4, 0)
    return blended


def make_cloud_overlay(last_rgb: np.ndarray, cloud_mask: np.ndarray) -> np.ndarray:
    """
    Overlay cloud regions (dark red tint) on the last input image.
    cloud_mask : (H, W) uint8
    """
    h, w = last_rgb.shape[:2]
    mask_resized = cv2.resize(cloud_mask, (w, h))
    overlay = last_rgb.copy()
    overlay[mask_resized > 127] = [
        int(overlay[mask_resized > 127, 0].mean() * 0.5),
        0,
        0,
    ]
    blended = cv2.addWeighted(last_rgb, 0.55, overlay, 0.45, 0)
    return blended


def ndarray_to_base64(img_array: np.ndarray, mode: str = "RGB") -> str:
    """Convert (H,W,3) or (H,W) numpy array → base64 PNG string for JSON."""
    if img_array.ndim == 2:
        pil = Image.fromarray(img_array.astype(np.uint8), mode="L")
    else:
        pil = Image.fromarray(img_array.astype(np.uint8), mode=mode)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Phase-3 classification ────────────────────────────────────────────────────

def classify_urban_growth(growth_pct: float, new_buildings: int):
    """
    Mirrors the classification shown on slide 31.
    """
    if growth_pct >= 10 or new_buildings >= 100:
        urban_class   = "High Urban Development"
        building_class = "High Building Growth"
    elif growth_pct >= 4 or new_buildings >= 30:
        urban_class   = "Moderate Urban Development"
        building_class = "High Building Growth"
    else:
        urban_class   = "Low / Stable Development"
        building_class = "Low Building Growth"
    return urban_class, building_class