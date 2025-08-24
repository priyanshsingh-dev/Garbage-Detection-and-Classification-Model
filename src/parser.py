import json
from collections import Counter
import matplotlib.pyplot as plt


#Load the JSON
with open("../data/annotations_clean.json", "r") as f:
    coco = json.load(f)


print("Keys:", coco.keys())  # Display the keys in the JSON file

images = coco["images"]
annotations= coco["annotations"]
categories = coco["categories"]

# map category IDs to names
cat_name= {c["id"]: c["name"] for c in categories}

# 5. Count how many boxes per category
counts = Counter(ann["category_id"] for ann in annotations)
for cat_id, name in cat_name.items():
    print(f"{name:20s}: {counts.get(cat_id,0)} instances")


img_id = images[8]["id"]
anns_for_first = [ann for ann in annotations if ann["image_id"] == img_id]
print(f"Image {img_id} has {len(anns_for_first)} boxes")


# Plotting the counts
super_map = {c["id"]: c["supercategory"] for c in categories}
super_counts = Counter(super_map[ann["category_id"]] for ann in annotations)

# Prepare data
names  = list(super_counts.keys())
values = [super_counts[name] for name in names]

# Plot
plt.figure(figsize=(16, 4))
plt.bar(names, values)
plt.xticks(rotation=30, ha='right')
plt.xlabel("Supercategory")
plt.ylabel("Number of Annotations")
plt.title("Annotations per Supercategory")
plt.tight_layout()
plt.show()