from ultralytics import YOLO
import os
import json
import torch

with open('settings.json') as f:
    s = json.load(f)

dataset_path = os.path.join(s['base_path'], s['dataset_yaml'])
if not os.path.exists(dataset_path):
    print(f"Error: {dataset_path} not found")
    exit(1)

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CPU mode")

model_path = os.path.join(s['base_path'], s['model_name']) if s['base_path'] else s['model_name']
model = YOLO(model_path)
model.train(
    data=dataset_path,
    epochs=s['epochs'],
    imgsz=s['image_size'],
    batch=s['batch_size'],
    name=s['model_output_name'],
    project=s['model_output_dir'],
    workers=s['workers'],
    amp=True,
    device=0 if torch.cuda.is_available() else 'cpu'
)

model_path = os.path.join(s['model_output_dir'], s['model_output_name'], 'weights', 'best.pt')
print(f"Model: {model_path}")

