# detector.py
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.4):
        # Ultralytics automatic download karega agar yolov8n.pt local nahi mila
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def detect(self, frame, allowed_classes=None):
        results = self.model(frame, conf=self.conf_thresh, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0])

            # Agar class filter lagaya ho
            if allowed_classes and cls_name not in allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_name))

        return detections