# main.py
import cv2
from detector import ObjectDetector
from tracker import ObjectTracker
from utils import Visualizer


def main():
    # 0 = Default Webcam, ya video path pass kar sakte hain (e.g., 'traffic.mp4')
    cap = cv2.VideoCapture(0)

    detector = ObjectDetector(model_path="yolov8n.pt", conf_thresh=0.4)
    tracker = ObjectTracker(max_age=30)
    vis = Visualizer()

    # Sirf specific classes track karne ke liye (None pass karein sabhi 80 classes ke liye)
    allowed_classes = ['person', 'car', 'bicycle', 'motorcycle']

    print("System shuru ho raha hai. Band karne ke liye 'q' press karein.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detection
        detections = detector.detect(frame, allowed_classes=allowed_classes)

        # 2. Tracking
        tracks = tracker.update(detections, frame)

        # 3. Annotation & FPS
        vis.draw_tracks(frame, tracks)
        vis.draw_fps(frame)

        # 4. Show output
        cv2.imshow("SmartTrack - Real-Time Object Detection & Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()