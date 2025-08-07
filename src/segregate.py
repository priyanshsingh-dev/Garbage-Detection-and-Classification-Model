import os, random, shutil
from pathlib import Path

# —— CONFIG ——————————————————————————————————————————————
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR      = PROJECT_ROOT / "data" / "Images"

DST_ROOT     = PROJECT_ROOT / "data" / "Images_split"
TRAIN_PCT    = 0.7
VAL_PCT      = 0.15
TEST_PCT     = 0.15
SEED         = 42


random.seed(42)

# 1. Create split folders
for split in ("train","val","test"):
    for cls in (p.name for p in SRC_DIR.iterdir() if p.is_dir()):
        (DST_ROOT / split / cls).mkdir(parents=True, exist_ok=True)

# 2. Walk each class
for cls_folder in SRC_DIR.iterdir():
    if not cls_folder.is_dir(): continue
    images = list(cls_folder.glob("*.*"))       # all files in this class
    random.shuffle(images)
    n = len(images)
    n_train = int(n * TRAIN_PCT)
    n_val   = int(n * VAL_PCT)

    splits = {
        "train": images[:n_train],
        "val":   images[n_train:n_train+n_val],
        "test":  images[n_train+n_val:]
    }

    for split, files in splits.items():
        for src_path in files:
            dst_path = DST_ROOT / split / cls_folder.name / src_path.name
            shutil.copy2(src_path, dst_path)

    print(f"{cls_folder.name}: "
          f"{len(splits['train'])} train, "
          f"{len(splits['val'])} val, "
          f"{len(splits['test'])} test")

print(f"\nDone! Splits created under {DST_ROOT}")
