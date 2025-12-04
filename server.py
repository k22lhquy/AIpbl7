from ultralytics import YOLO
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Cho phép gọi từ frontend

# Load Models
model_human = YOLO("bestHuman.pt")
model_acid  = YOLO("bestACID.pt")
model_hole  = YOLO("bestHole.pt")


def detect_objects(image):
    results_human = model_human(image)[0]
    results_acid  = model_acid(image)[0]
    results_hole  = model_hole(image)[0]

    detections = {
        "human": [],
        "acid": [],
        "hole": []
    }

    # ========== HUMAN ==========
    for box in results_human.boxes:
        cls_id = int(box.cls[0])
        class_name = model_human.names[cls_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections["human"].append({
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": float(box.conf[0]),
            "x1": int(x1), "y1": int(y1),
            "x2": int(x2), "y2": int(y2)
        })

    # ========== ACID ==========
    for box in results_acid.boxes:
        cls_id = int(box.cls[0])
        class_name = model_acid.names[cls_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections["acid"].append({
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": float(box.conf[0]),
            "x1": int(x1), "y1": int(y1),
            "x2": int(x2), "y2": int(y2)
        })

    # ========== HOLE ==========
    for box in results_hole.boxes:
        cls_id = int(box.cls[0])
        class_name = model_hole.names[cls_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections["hole"].append({
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": float(box.conf[0]),
            "x1": int(x1), "y1": int(y1),
            "x2": int(x2), "y2": int(y2)
        })

    return detections


# --------------------------------------------------
# API chính: nhận ảnh → trả JSON detect
# --------------------------------------------------
@app.route("/detect", methods=["POST"])
def detect_api():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    # Convert bytes → numpy array
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    detections = detect_objects(img)
    return jsonify(detections)


# --------------------------------------------------
# API test: GET → trả JSON mẫu
# --------------------------------------------------
@app.route("/test", methods=["GET"])
def test_api():
    sample_json = {
        "human": [
            {"class_id": 0, "class_name": "person", "confidence": 0.95, "x1": 100, "y1": 50, "x2": 300, "y2": 450},
            {"class_id": 1, "class_name": "helmet", "confidence": 0.88, "x1": 150, "y1": 30, "x2": 250, "y2": 100}
        ],
        "acid": [
            {"class_id": 0, "class_name": "acid_bottle", "confidence": 0.87, "x1": 300, "y1": 150, "x2": 380, "y2": 230}
        ],
        "hole": [
            {"class_id": 0, "class_name": "hole", "confidence": 0.95, "x1": 500, "y1": 320, "x2": 560, "y2": 390}
        ]
    }
    return jsonify(sample_json)


# --------------------------------------------------
# Run Flask trên Render
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

