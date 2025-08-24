from ultralytics import YOLO

# load and train the model
model=YOLO("yolov8n.pt")
result=model.train(data = "../data/data.yaml", epochs = 100, imgsz = 640, workers = 0)
