import numpy as np
from PIL import Image
import io, math
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# ── helpers ──────────────────────────────────────────────────────────────────

def tif_to_rgb_array(tif_bytes: bytes) -> np.ndarray:
    """Convert raw .tif bytes → (H, W, 3) uint8 numpy array."""
    from PIL import Image
    img = Image.open(io.BytesIO(tif_bytes)).convert("RGB")
    return np.array(img)


def estimate_building_count(img: np.ndarray) -> int:
    """
    Lightweight proxy for building count:
    threshold bright pixels (buildings appear bright in RGB composites),
    then count connected blobs via a simple scan.
    Replace this with your own mask-based counter if you have JSON annotations.
    """
    gray = img.mean(axis=2)
    thresh = gray > 180
    # count runs of True pixels row-wise as a proxy
    count = int(thresh.sum() // 120)
    return max(count, 0)


def estimate_area_acres(img: np.ndarray, pixel_size_m: float = 0.5) -> float:
    """
    Estimate built-up area in acres from bright-pixel count.
    SpaceNet-7 WorldView-3 native GSD ≈ 0.3 m; mosaics are often 0.5 m.
    """
    gray = img.mean(axis=2)
    built_pixels = int((gray > 180).sum())
    area_m2 = built_pixels * (pixel_size_m ** 2)
    area_acres = area_m2 / 4046.856
    return round(area_acres, 4)


def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8]:
        return "Monsoon"
    else:
        return "Autumn"


# ── Phase-1 main entry ────────────────────────────────────────────────────────

def run_comparative_analysis(images_rgb: list[np.ndarray], start_year: int = 2018, start_month: int = 1):
    """
    images_rgb : list of (H,W,3) arrays ordered oldest → newest
    Returns a dict with all Phase-1 metrics.
    """
    n = len(images_rgb)
    records = []

    base_date = datetime(start_year, start_month, 1)

    counts = [estimate_building_count(img) for img in images_rgb]
    areas  = [estimate_area_acres(img)     for img in images_rgb]

    for i in range(1, n):
        prev_date = base_date + relativedelta(months=i - 1)
        curr_date = base_date + relativedelta(months=i)
        new_bldg  = max(counts[i] - counts[i - 1], 0)
        area_chg  = round(areas[i] - areas[i - 1], 4)

        records.append({
            "start_date":          prev_date.strftime("%Y-%m-%d"),
            "end_date":            curr_date.strftime("%Y-%m-%d"),
            "prev_building_count": counts[i - 1],
            "curr_building_count": counts[i],
            "new_building_count":  new_bldg,
            "prev_area_acres":     areas[i - 1],
            "curr_area_acres":     areas[i],
            "area_change_acres":   area_chg,
            "season":              get_season(curr_date.month),
        })

    # seasonal aggregation
    seasonal: dict[str, list] = {"Winter": [], "Summer": [], "Monsoon": [], "Autumn": []}
    for r in records:
        seasonal[r["season"]].append(r["new_building_count"])
    seasonal_avg = {k: round(float(np.mean(v)), 2) if v else 0.0 for k, v in seasonal.items()}

    # overall summary
    total_new   = sum(r["new_building_count"] for r in records)
    total_area  = round(areas[-1] - areas[0], 4)

    return {
        "monthly_records": records,
        "seasonal_avg":    seasonal_avg,
        "total_new_buildings": total_new,
        "total_area_change_acres": total_area,
        "baseline_area_acres":    areas[0],
        "final_area_acres":       areas[-1],
        "baseline_building_count": counts[0],
        "final_building_count":   counts[-1],
        "num_images": n,
    }