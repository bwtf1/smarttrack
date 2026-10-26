# utils.py
import cv2
import time


class Visualizer:
    def __init__(self):
        self.prev_time = time.time()

    def draw_tracks(self, frame, tracks):
        for track_id, ltrb, cls_name in tracks:
            x1, y1, x2, y2 = map(int, ltrb)

            # Bounding box draw karna
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label background aur text
            label = f"{cls_name} | ID: {track_id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 22), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    def draw_fps(self, frame):
        curr_time = time.time()
        fps = 1.0 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)