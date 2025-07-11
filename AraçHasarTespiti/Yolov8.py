import os
import gdown
import cv2
import json
import numpy as np
from ultralytics import YOLO

# === MODEL AYARLARI ===
MODEL_ID = "1wlqODfvem4orqlkxcBTiGAGoMVZbp5Or"
model_path = "bestt.pt"

# === MODEL İNDİRME ===
if not os.path.exists(MODEL_PATH):
    print("📥 YOLOv8 modeli indiriliyor...")
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

try:
    model = YOLO(model_path)
    print("Model başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")
    exit()

output_dir = "outputYoloV8"
os.makedirs(output_dir, exist_ok=True)

def predict(image_path):
    """
    Verilen görsel yoluyla YOLOv8 modeliyle tahmin yapar.
    JSON formatında tahmin sonuçları ve anotasyonlu OpenCV görseli döner.
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError("Görsel yolu bulunamadı.")

        # Görseli oku
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Görsel okunamadı veya bozuk.")
        orig_h, orig_w = img.shape[:2]

        # Model tahmini (YOLOv8 input olarak kendi içinde 640x640'e resize eder)
        results = model(image_path)

        detections = []

        for result in results:
            # Tahmin sırasında kullanılan model giriş boyutu (imgsz)
            input_h, input_w = result.orig_shape
            scale_x = orig_w / input_w
            scale_y = orig_h / input_h

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Kutuları geri ölçekle
                x1 *= scale_x
                y1 *= scale_y
                x2 *= scale_x
                y2 *= scale_y

                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = model.names[cls_id] if cls_id < len(model.names) else f"Class {cls_id}"

                detections.append({
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })

                # Orijinal görsele kutu çiz
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(img, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # JSON + Annotated görsel döndür
        return json.dumps(detections, indent=4, ensure_ascii=False), img

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=4, ensure_ascii=False), None
