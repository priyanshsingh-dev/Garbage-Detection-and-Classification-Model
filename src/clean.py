import os, json
import pandas as pd

#CONFIG
PROJECT_DIR   = r"C:\Users\ACER\Desktop\Trash Detection And Classification"
DATA_DIR      = os.path.join(PROJECT_DIR, "data")
ANN_IN        = os.path.join(DATA_DIR, "annotations.json")
ANN_OUT       = os.path.join(PROJECT_DIR, "annotations_clean.json")
IMG_ROOT      = os.path.join(DATA_DIR, "Images")

MIN_W, MIN_H  = 5, 5        
MIN_AREA      = 100         
MAX_IMG_COVER = 0.90        # 90% of image area

if __name__ == "__main__":
    # 1. Load JSON to DataFrames
    with open(ANN_IN, "r") as f:
        coco = json.load(f)

    imgs_df = pd.DataFrame(coco["images"])
    ann_df  = pd.DataFrame(coco["annotations"])
    cat_df  = pd.DataFrame(coco["categories"])

    # 2. Expand bbox into columns
    ann_df[["x","y","w","h"]] = pd.DataFrame(ann_df.bbox.tolist(), index=ann_df.index)

    # 3. Merge image dims
    df = (ann_df
        .merge(imgs_df[["id","file_name","width","height"]],
                left_on="image_id", right_on="id", suffixes=("","_img"))
        )

    # 4. Compute areas & ratios
    df["box_area"] = df["w"] * df["h"]
    df["img_area"] = df["width"] * df["height"]
    df["area_ratio"] = df["box_area"] / df["img_area"]

    # 5. Check file existence
    df["exists"] = df["file_name"].apply(
        lambda fn: os.path.isfile(os.path.join(IMG_ROOT, fn))
    )

    # 6. Apply cleaning masks
    mask = (
        (df.w  >= MIN_W) &
        (df.h  >= MIN_H) &
        (df.box_area >= MIN_AREA) &
        (df.area_ratio <= MAX_IMG_COVER) &
        (df.x  >= 0) &
        (df.y  >= 0) &
        (df.x + df.w <= df.width) &
        (df.y + df.h <= df.height) &
        (df.exists)
    )

    clean_df = df[mask].copy()

    # 7. Drop images with no remaining boxes
    valid_image_ids = clean_df["image_id"].unique()
    clean_imgs = imgs_df[imgs_df.id.isin(valid_image_ids)].to_dict("records")

    # 8. Clean annotations (drop the extra columns)
    clean_anns = (
        clean_df
        .drop(columns=["x","y","w","h","box_area","img_area","area_ratio","exists",
                    "file_name","width","height","id"])
        .to_dict("records")
    )

    # 9. prune categories with no instances left
    valid_cat_ids = clean_df["category_id"].unique()
    clean_cats = [c for c in coco["categories"] if c["id"] in valid_cat_ids]

    # 10. Write cleaned COCO JSON
    clean_coco = {
        "info":        coco.get("info", {}),
        "licenses":    coco.get("licenses", []),
        "images":      clean_imgs,
        "annotations": clean_anns,
        "categories":  clean_cats
    }
    with open(ANN_OUT, "w") as f:
        json.dump(clean_coco, f, indent=2)


    # This Tells you the amount of data left after cleaning and also tels you the number of categories left that you would need for your data.yaml file

    print(f"Kept {len(clean_imgs)} images, {len(clean_anns)} annotations, "
        f"{len(clean_cats)} categories")

    with open("count.txt", "w") as f:
        f.write(f"{len(clean_cats)}")
        


    
    
