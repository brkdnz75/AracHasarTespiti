import os
import gdown
import torch
import cv2
import json
import numpy as np

# === MODEL AYARLARI ===
MODEL_ID = "17dVBQN5LZnUpbtJIji3KhTot_Ks3Ju6G"  # ← senin Google Drive ID
model_path = "best.pt"

# === MODEL İNDİRME ===
if not os.path.exists(MODEL_PATH):
    print("📥 YOLOv5 modeli indiriliyor...")
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Modeli yükle
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False, verbose=True)
    print("Model başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")
    exit()

# Çıktı klasörü oluştur
output_dir = "outputYoloV5"
os.makedirs(output_dir, exist_ok=True)

def predict(image_path):
    """
    Verilen görsel yoluyla tahmin yapar.
    JSON formatında tahmin sonuçları ve anotasyonlu OpenCV görseli döner.
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError("Görsel yolu bulunamadı.")

        # Görseli oku
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Görsel okunamadı veya bozuk.")

        # Orijinal boyutları sakla
        orig_h, orig_w = img.shape[:2]

        # Model için yeniden boyutlandır (416x416)
        resized_img = cv2.resize(img, (416, 416))

        # Model ile tahmin
        results = model(resized_img)

        # Kutuları orijinal boyuta ölçekle
        scale_x = orig_w / 416
        scale_y = orig_h / 416

        detections = []
        for *box, conf, cls_id in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = box

            # Orijinal boyuta çevir
            x1 *= scale_x
            y1 *= scale_y
            x2 *= scale_x
            y2 *= scale_y

            conf = float(conf)
            cls_id = int(cls_id)
            cls_name = model.names[cls_id] if cls_id < len(model.names) else f"Class {cls_id}"

            detections.append({
                "class": cls_name,
                "confidence": conf,
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })

            # Kutuları çiz
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(img, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return json.dumps(detections, indent=4, ensure_ascii=False), img

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=4, ensure_ascii=False), None