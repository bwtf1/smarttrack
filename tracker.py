# tracker.py
from deep_sort_realtime.deepsort_tracker import DeepSort


class ObjectTracker:
    def __init__(self, max_age=30):
        self.tracker = DeepSort(max_age=max_age)

    def update(self, detections, frame):
        # Deep SORT format: [([left, top, w, h], confidence, detection_class)]
        tracks = self.tracker.update_tracks(detections, frame=frame)
        active_tracks = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            cls_name = track.get_det_class()
            active_tracks.append((track_id, ltrb, cls_name))

        return active_tracks