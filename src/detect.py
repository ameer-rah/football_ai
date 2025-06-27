import os
from ultralytics import YOLO
import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  
RAW_DIR = os.path.join(ROOT_DIR, 'data/raw')
PROCESSED_DIR = os.path.join(ROOT_DIR, 'data/processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)
model = YOLO('runs/detect/train7/weights/best.pt')

for filename in os.listdir(RAW_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(RAW_DIR, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {img_path}")
            continue

        results = model(img, conf=0.1)        
        print(results[0].boxes)
        save_path = os.path.join(PROCESSED_DIR, f'detected_{filename}')
        results[0].save(filename=save_path)
        print(f"Processed and saved: {save_path}")

print("Detection complete. Add images to the data/raw/ folder and rerun this script as needed.") 