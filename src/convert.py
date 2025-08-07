
import os
import json
from pathlib import Path

# —— CONFIG ——————————————————————————————————————————————
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMG_ROOT     = PROJECT_ROOT/"data"/"Images_Split"
JSON_PATH    = PROJECT_ROOT/"data"/"annotations_clean.json"
LBL_ROOT     = PROJECT_ROOT/"data"/"labels_yolo_split"
# ————————————————————————————————————————————————————————

# 1. Load the cleaned COCO JSON
with open(JSON_PATH, "r") as f:
    coco = json.load(f)

# 2. Build a mapping: file_name (relative path in JSON) → image info dict
file2info = { img["file_name"]: img for img in coco["images"] }

# 3. Define your 8 coarse classes and sub-mappings (as before)
COARSE = [
    "Bottle","Carton","Bottle cap","Can","Cup",
    "Plastic bag & wrapper","Other plastic","Cigarette",
]
coarse_to_fine = {
    "Bottle": ["Other plastic bottle","Clear plastic bottle","Glass bottle"],
    "Carton": ["Other carton","Egg carton","Drink carton","Corrugated carton","Meal carton","Pizza box"],
    "Bottle cap": ["Plastic bottle cap","Metal bottle cap"],
    "Can": ["Drink can","Food Can","Aerosol"],
    "Cup": ["Paper cup","Disposable plastic cup","Foam cup","Glass cup","Other plastic cup"],
    "Plastic bag & wrapper": ["Single-use carrier bag","Polypropylene bag","Garbage bag","Other plastic wrapper","Crisp packet","Plastic film"],
    "Other plastic": ["Other plastic","Other plastic container","Plastic utensils","Plastic glooves","Tupperware","Disposable food container","Foam food container","Spread tub"],
    "Cigarette": ["Cigarette"],
}
# name→id
name2id = {c["name"]: c["id"] for c in coco["categories"]}
# id→newIndex
cat2new = {}
for new_idx, coarse in enumerate(COARSE):
    for fine in coarse_to_fine[coarse]:
        orig_id = name2id.get(fine)
        if orig_id is None:
            raise ValueError(f"Category '{fine}' not in JSON")
        cat2new[orig_id] = new_idx

# 4. Prepare label directories
for split in ("train","val","test"):
    for coarse in COARSE:
        (LBL_ROOT/split/coarse.replace(" ", "_")).mkdir(parents=True, exist_ok=True)

# 5. Iterate through split images
for split in ("train","val","test"):
    img_dir = IMG_ROOT/split
    for img_path in img_dir.rglob("*.*"):
        # compute its relative path as in JSON, e.g. "batch_3/IMG_5057.JPG"
        rel = img_path.relative_to(IMG_ROOT)
        rel_str = str(rel).replace("\\","/")  # match JSON’s forward slashes

        info = file2info.get(rel_str)
        if info is None:
            # no matching JSON entry—skip
            continue

        img_id = info["id"]
        w_img, h_img = info["width"], info["height"]

        # collect all annotations for this image
        lines = []
        for ann in coco["annotations"]:
            if ann["image_id"] != img_id:
                continue
            orig_id = ann["category_id"]
            if orig_id not in cat2new:
                continue
            class_idx = cat2new[orig_id]
            x,y,bw,bh = ann["bbox"]
            # convert to YOLO format
            x_c = (x + bw/2) / w_img
            y_c = (y + bh/2) / h_img
            bw_n = bw / w_img
            bh_n = bh / h_img
            lines.append(f"{class_idx} {x_c:.6f} {y_c:.6f} {bw_n:.6f} {bh_n:.6f}")

        # only write if there are relevant boxes
        if lines:
            lbl_path = LBL_ROOT/split/rel.parent.name/f"{img_path.stem}.txt"
            with open(lbl_path, "w") as out:
                out.write("\n".join(lines))

print("COCO → YOLO conversion complete.")
