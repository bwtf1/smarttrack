# SmartTrack: Detect, Identify, Track

### Real-Time Object Detection and Multi-Object Tracking

SmartTrack is a computer vision project designed to detect and track multiple objects in video frames. It uses a pre-trained YOLO model for object detection and a tracking algorithm such as SORT or Deep SORT to maintain object identities across consecutive frames.

## Project Information

- **Project Name:** SmartTrack: Detect, Identify, Track
- **Project Title:** Real-Time Object Detection and Multi-Object Tracking
- **Group Name:** TECH ELITE
- **Group Number:** 4
- **Department:** Computer Science and Engineering (Artificial Intelligence and Machine Learning)
- **Institution:** ITS Engineering College
- **Academic Year:** 2026–2027
- **Faculty Guide:** **Ms. Anjali Srivastava**

## Team Members

| S. No. | Team Member Name | Roll Number |
|---|---|---|
| 1 | Rishav Raj | 2402221530102 |
| 2 | Rahul Raj Yadav | 2402221530096 |
| 3 | Shashank Kumar | 2402221530116 |
| 4 | Ravi Kumar Singh | 2402221530098 |

## Problem Statement

Manual monitoring of video feeds in traffic, campus, retail, and security environments is time-consuming and prone to human error. Detecting objects in individual frames does not maintain their identities across time. Without persistent tracking IDs, the same object may be counted multiple times, making movement analysis difficult.

SmartTrack aims to combine object detection and multi-object tracking to identify objects and maintain their identities across consecutive video frames.

## Objectives

- Capture video from a webcam or video file using OpenCV.
- Detect objects using a pre-trained YOLO model.
- Track multiple objects using SORT or Deep SORT.
- Display bounding boxes, class labels, confidence scores, and tracking IDs.
- Display the processing frame rate (FPS), where implemented.
- Support optional object counting, movement trails, video saving, and CSV logging.

## Key Features

The project blueprint describes these intended features; their availability depends on the actual implementation.

- Webcam and video-file input.
- Detection of common object categories supported by the COCO-trained model.
- Bounding boxes and object class labels.
- Confidence scores for detections.
- Persistent tracking IDs.
- FPS monitoring.
- Optional confidence threshold and class filtering.
- Optional object counting and movement trails.
- Optional saving of annotated videos and detection logs.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Video capture and frame processing |
| YOLOv8 | Object detection |
| PyTorch | Deep learning framework |
| SORT / Deep SORT | Multi-object tracking |
| NumPy | Numerical processing |
| SciPy and FilterPy | Supporting libraries for SORT, if used |
| deep-sort-realtime | Supporting library for Deep SORT, if used |
| Streamlit / Tkinter | Optional user interface |

## System Workflow

1. Capture video from a webcam or video file.
2. Read and prepare video frames using OpenCV.
3. Detect objects using the YOLO model.
4. Filter detections using confidence thresholds and non-maximum suppression.
5. Associate detections across frames using SORT or Deep SORT.
6. Assign and maintain tracking IDs.
7. Display annotated frames with bounding boxes, labels, and IDs.
8. Save output video or detection logs if those features are implemented.

**Workflow:** Video Input → OpenCV → YOLO Detection → Detection Filtering → Object Tracking → Annotated Video Output

## Installation

Install Python 3.9 or later.

Create and activate a virtual environment.

**Windows Command Prompt:**

```bat
python -m venv .venv
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install ultralytics opencv-python numpy scipy filterpy deep-sort-realtime
```

Install only the libraries required by your selected tracker. If the project includes a `requirements.txt` file, you can install dependencies using:

```bash
pip install -r requirements.txt
```

## Running the Project

The command depends on the main Python file in your repository.

For example, if your entry-point file is named `main.py`:

```bash
python main.py
```



## Dataset

The basic system uses pre-trained YOLO weights associated with the COCO dataset. Training a model from scratch is not required for detecting the supported classes.

Possible video inputs include:

- Live webcam footage.
- Recorded traffic videos.
- Pedestrian and campus videos.
- Public multi-object tracking datasets such as MOT16 and MOT17 for additional evaluation.

## Evaluation

The system may be evaluated using the following criteria:

- Correctness of object detection.
- Consistency of tracking IDs across frames.
- Tracking behaviour when objects overlap or become temporarily hidden.
- Processing speed measured in FPS.
- Tracking metrics such as MOTA and IDF1, when suitable ground-truth annotations are available.

The original project blueprint identifies approximately 15–30 FPS as a target. Actual performance depends on the model, hardware, video resolution, and tracking configuration and should be measured during testing.

## Team Work Distribution

- **Rishav Raj — Project Lead:** Environment setup, video input module, module integration, and GitHub management.
- **Rahul Raj Yadav — Object Detection:** YOLO model loading, inference, confidence filtering, NMS, and detection testing.
- **Shashank Kumar — Tracking Module:** SORT/Deep SORT integration, ID assignment, and tracking parameter tuning.
- **Ravi Kumar Singh — Visualization and UI:** Bounding boxes, labels, tracking IDs, FPS display, interface controls, documentation, and report preparation.

Testing, performance evaluation, presentation, and report preparation are shared responsibilities.

## Results and Screenshots

Add screenshots of the actual running application to this section.

Suggested screenshots:

1. Application interface.
2. Object detection with bounding boxes.
3. Multiple objects with their tracking IDs.
4. FPS counter or other implemented functionality.

Example for adding a screenshot stored in the repository:

```markdown
![SmartTrack Detection Output](screenshots/smarttrack-output.png)
```

Ensure the image path matches the actual file in your repository. Only report results that have been verified on the running application.

## Future Improvements

- Improve tracking in crowded environments.
- Add object-counting and line-crossing analytics.
- Support saving annotated videos.
- Export detection logs to CSV.
- Evaluate tracking using labelled benchmark datasets.
- Fine-tune the detector for specific object categories if required.

## Acknowledgement

We sincerely thank our faculty guide, **Ms. Anjali Srivastava**, for her guidance and support throughout the SmartTrack project. We also acknowledge the contributions of all members of **TECH ELITE (Group 4)**.

## License

No license has been specified for this project. Add an appropriate license file if you intend to distribute the project for reuse.

---

**Developed by:** TECH ELITE — Group 4  
**Faculty Guide:** Ms. Anjali Srivastava  
**Institution:** ITS Engineering College  
**Academic Year:** 2026–2027
