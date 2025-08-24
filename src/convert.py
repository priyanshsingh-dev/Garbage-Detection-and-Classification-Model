import json
import os
from sklearn.model_selection import train_test_split
import shutil



json_path= '../data/annotations_clean.json'
image_root_dir = '../data/images'

def convert_taco_to_yolo(json_path, image_root_dir, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels/val'), exist_ok=True)

    image_ids = [img['id'] for img in images]
    train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)

    image_id_to_filename = {img['id']: img['file_name'] for img in images}
    category_id_to_name = {cat['id']: cat['name'] for cat in categories}

    for img in images:
        img_id = img['id']
        filename = img['file_name']
        img_width = img['width']
        img_height = img['height']

        if img_id in train_ids:
            image_dir = os.path.join(output_dir, 'images/train')
            label_dir = os.path.join(output_dir, 'labels/train')
        else:
            image_dir = os.path.join(output_dir, 'images/val')
            label_dir = os.path.join(output_dir, 'labels/val')

        full_image_path = os.path.join(image_root_dir, filename)

        if os.path.exists(full_image_path):
            shutil.copy(full_image_path, os.path.join(image_dir, filename.split('/')[-1]))
        else:
            print(f"Warning: Image {full_image_path} not found.")
            continue

        label_file = os.path.join(label_dir, filename.split('/')[-1].replace('.jpg', '.txt').replace('.JPG', '.txt'))

        with open(label_file, 'w') as lf:
            for ann in annotations:
                if ann['image_id'] == img_id:
                    category_id = ann['category_id']
                    bbox = ann['bbox']
                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width = bbox[2] / img_width
                    height = bbox[3] / img_height
                    lf.write(f"{category_id} {x_center} {y_center} {width} {height}\n")

output_yolo_dir ='../data/image_split'

convert_taco_to_yolo(json_path,image_root_dir, output_yolo_dir)
count_categories = int(open('count.txt', 'r').read().strip())
# Create data.yaml
with open('../data/data.yaml', 'w') as f:
    f.write(f"path: image_split\n")
    f.write(f"train: images/train\n")
    f.write(f"val: images/val\n")
    f.write(f"nc: {count_categories}\n") 
    categories = json.load(open('../data/annotations_clean.json', 'r'))['categories']
    f.write("names: " + str([cat['name'] for cat in categories]) + "\n")
