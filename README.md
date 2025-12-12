# Powerline Component Detector

YOLO11-based detector for powerline components: pylon, conductor, insulator, pylon_fissure

## Structure

```
powerline/
├── train_yolo.py          # Training
├── inference_and_export.py # Testing
├── show_metrics.py        # Display metrics
├── settings_default.json   # Template config
├── settings.json          # All paths (copy from settings_default.json)
├── dataset.yaml           # YOLO config
├── pixi.toml              # Environment
├── pixi.lock              # Exact versions
├── .gitignore             # Minimal
├── dataset/               # 36 train + 9 val
├── images_test/           # 600 images
├── detected_defects/      # Images with pylon_fissure detected
├── documents/             # Documentation files
└── models/                # Trained models and pretrained weights
```

## Usage

1. Install environment: `pixi install`
2. Train model: `pixi run python train_yolo.py`
3. View metrics: `pixi run python show_metrics.py` (generates `metrics.txt`)
4. Test on images: `pixi run python inference_and_export.py`
   - Generates `detections.json` and `detections.csv`
   - Copies images with `pylon_fissure` to `detected_defects/`

## Configuration

1. Copy `settings_default.json` to `settings.json`
2. Edit `settings.json` and set `base_path` to your project directory (e.g., `C:\\Users\\YourName\\Documents\\powerline` on Windows or `/home/username/powerline` on Linux)

## Dataset

The dataset contains 45 annotated images (36 for training, 9 for validation):
- Training images: `dataset/train/images/`
- Training labels: `dataset/train/labels/`
- Validation images: `dataset/val/images/`
- Validation labels: `dataset/val/labels/`

Annotations were created using **LabelImg** in **YOLO format** (one `.txt` file per image with normalized coordinates).

## Metrics

After training, generate metrics file:
```bash
pixi run python show_metrics.py
```

This creates `metrics.txt`


## Output Files

- `metrics.txt`: Training metrics (precision, recall, mAP)
- `detections.json`: All detections in JSON format
- `detections.csv`: All detections in CSV format
- `detected_defects/`: Images where `pylon_fissure` was detected
- `models/powerline_detector/weights/best.pt`: Best trained model

