import os, json, random
import cv2
from matplotlib import pyplot as plt

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))   
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                

# 2. Point to your data
IMG_DIR  = os.path.join(PROJECT_ROOT, "data", "Images")     
ANN_PATH = os.path.join(PROJECT_ROOT,"data", "annotations_clean.json") 

# load annotations
with open(ANN_PATH,"r")as f:
    coco=json.load(f)

imgs = {img["id"]: img for img in coco["images"]}
anns = coco["annotations"]
cats = {c["id"]: c["name"] for c in coco["categories"]}

# anns to image id
ans_image= {}
for a in anns:
    ans_image.setdefault(a["image_id"], []).append(a)

sample_ids = random.sample(list(ans_image.keys()),k=5)

for img_id in sample_ids:
    info = imgs[img_id]
    img_path = os.path.join(IMG_DIR, info["file_name"])
    img=cv2.imread(img_path)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

for a in ans_image[img_id]:
    x, y, w, h = map(int, a["bbox"])
    label = cats[a["category_id"]]
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(img, label, (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 5, (255,0,0), 2)
    

# Plot 
plt.figure(figsize=(6,8))
plt.imshow(img)
plt.title(info["file_name"])
plt.axis("off")
plt.show()