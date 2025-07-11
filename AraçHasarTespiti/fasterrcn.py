import os
import gdown
import cv2
import torch
import json
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# === Google Drive'dan Model İndirme ===
MODEL_ID = "1DS4fuEhgjxW6ZDUYcFZtlpERvBiGVlZu"  # ← senin Drive ID
model_path = "model_final.pth"

if not os.path.exists(model_path):
    print("📥 Model indiriliyor...")
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    import gdown
    gdown.download(url, model_path, quiet=False)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # Sınıf sayısı
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

# === SINIF İSİMLERİ ===
class_names = ["Bumper", "Fender", "Light", "Windshield", "Dickey", "Door", "Hood"]

# === ÇIKTI KLASÖRÜ ===
output_dir = "output_detectron2"
os.makedirs(output_dir, exist_ok=True)

def predict(image_path):
    """
    Verilen görsel yoluyla tahmin yapar.
    JSON formatında tahmin sonuçları ve OpenCV anotasyonlu görsel döner.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Görsel okunamadı.")

        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")

        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()

        detections = []
        for box, score, cls_id in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"

            detections.append({
                "class": cls_name,
                "confidence": float(score),
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })

            # Anotasyon çizimleri
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            label = f"{cls_name} {score:.2f}"
            cv2.putText(img, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Opsiyonel olarak kayıt (yorum satırı olarak bırakıldı)
        # name = os.path.splitext(os.path.basename(image_path))[0]
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # annotated_path = os.path.join(output_dir, f"{name}_tahmin_{timestamp}.jpg")
        # json_path = os.path.join(output_dir, f"{name}_tahmin_{timestamp}.json")
        # cv2.imwrite(annotated_path, img)
        # with open(json_path, "w", encoding="utf-8") as f:
        #     json.dump(detections, f, indent=4, ensure_ascii=False)

        return json.dumps(detections, indent=4, ensure_ascii=False), img

    except Exception as e:
        return json.dumps({"error": str(e)}), None


if __name__ == "__main__":
    while True:
        image_path = input("Görsel yolu (Çıkmak için 'çıkış' yazın): ").strip()
        if image_path.lower() == "çıkış":
            break
        if not os.path.exists(image_path):
            print("Görsel bulunamadı. Tekrar deneyin.")
            continue
        json_result, annotated_img = predict(image_path)
        print("Tahmin sonucu (JSON):")
        print(json_result)

        if annotated_img is not None:
            plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title("Faster R-CNN Tespitleri")
            plt.show()
        else:
            print("Görsel işlenemedi.")



