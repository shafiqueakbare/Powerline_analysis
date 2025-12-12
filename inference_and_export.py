from ultralytics import YOLO
import os
import json
import csv
import glob
import shutil

with open('settings.json') as f:
    s = json.load(f)

base = s['base_path']
model_dir = s['model_output_dir']
model_name = s['model_output_name']
model_path = os.path.join(model_dir, model_name, 'weights', 'best.pt')
test_dir = os.path.join(base, s['test_images_dir'])
out_json = os.path.join(base, s['output_json'])
out_csv = os.path.join(base, s['output_csv'])
defects_dir = os.path.join(base, 'detected_defects')

classes = {0: "pylon", 1: "conductor", 2: "insulator", 3: "pylon_fissure"}

if not os.path.exists(model_path):
    model_path = os.path.join(model_dir, model_name, 'weights', 'last.pt')
    if not os.path.exists(model_path):
        print("Error: Model not found. Run train_yolo.py first")
        exit(1)

model = YOLO(model_path)

images = []
for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
    images.extend(glob.glob(os.path.join(test_dir, ext)))

# Remove duplicates (Windows is case-insensitive)
images = list(set(images))

print(f"Processing {len(images)} images...")

# Create defects directory
os.makedirs(defects_dir, exist_ok=True)

results = []
images_with_fissure = set()
for i, img in enumerate(images, 1):
    if i % 50 == 0:
        print(f"{i}/{len(images)}")
    
    pred = model.predict(img, conf=s['confidence_threshold'], verbose=False)
    img_name = os.path.basename(img)
    
    for r in pred:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            
            results.append({
                "image": img_name,
                "class": classes[cls],
                "class_id": cls,
                "score": round(conf, 4),
                "bbox": {
                    "x1": round(float(x1), 2),
                    "y1": round(float(y1), 2),
                    "x2": round(float(x2), 2),
                    "y2": round(float(y2), 2)
                }
            })
            
            # Track images with pylon_fissure
            if cls == 3:  # pylon_fissure
                images_with_fissure.add(img)

with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)

with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["image", "class", "class_id", "score", "x1", "y1", "x2", "y2"])
    for r in results:
        w.writerow([r["image"], r["class"], r["class_id"], r["score"],
                   r["bbox"]["x1"], r["bbox"]["y1"], r["bbox"]["x2"], r["bbox"]["y2"]])

# Copy images with pylon_fissure
for img_path in images_with_fissure:
    img_name = os.path.basename(img_path)
    dest_path = os.path.join(defects_dir, img_name)
    shutil.copy2(img_path, dest_path)

print(f"Done: {len(results)} detections -> {out_json} and {out_csv}")
print(f"Copied {len(images_with_fissure)} images with pylon_fissure to {defects_dir}")

