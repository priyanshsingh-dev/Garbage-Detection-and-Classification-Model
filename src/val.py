from ultralytics import YOLO

model = YOLO("runs/detect/train7/weights/best.pt")  # fix path
VAL_DIR = "../data/image_split/images/val"          # fix folder name

results = model.predict(
    source=VAL_DIR,
    conf=0.25,           # adjust threshold
    save=True,           # saves annotated images
    save_txt=False,      # set True to save YOLO txt predictions
    project="runs/predict",
    name="trash_val",
    device=0           
)
print("Saved to:", results[0].save_dir)
