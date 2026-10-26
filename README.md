# SmartTrack: Real-Time Object Detection and Tracking

## Project Overview

SmartTrack is a Python-based computer vision system designed to detect, identify, and track multiple objects in real time from live camera feeds or recorded video sources. The project uses modern deep learning techniques for object detection and tracking, with a focus on real-world monitoring applications such as traffic analysis, campus surveillance, security systems, and general-purpose object tracking.

The system combines YOLOv8 for object detection and DeepSORT for multi-object tracking, enabling persistent object identities across consecutive frames. It also provides a user-friendly interface and analytics dashboard for monitoring results efficiently.

## Project Title

SmartTrack: Detect, Identify, Track

## Institution

ITS Engineering College

## Department

Computer Science and Engineering (Artificial Intelligence and Machine Learning)

## Faculty Guide

Anjali Shrivastav

## Team Members

- Shashank Kumar
- Ravi Kumar Singh
- Rishav Raj
- Rahul Raj

## Abstract

The increasing demand for intelligent monitoring systems has led to the growth of computer vision-based solutions for object detection and tracking. Traditional surveillance often relies on manual observation, which is inefficient, time-consuming, and highly prone to error. SmartTrack addresses this limitation by automating the detection of objects and maintaining their identities across video frames in real time.

This project implements a robust object tracking workflow that captures frames from a webcam or uploaded videos, detects objects using a pretrained YOLO model, and tracks them using a deep learning-based tracking algorithm. The system highlights detected objects using bounding boxes, labels, and unique tracking IDs, making it suitable for use in traffic monitoring, pedestrian tracking, object counting, and monitoring-related applications.

## Problem Statement

Monitoring dynamic scenes manually is an effort-intensive task and is often inaccurate when the environment is crowded or fast-moving. Many surveillance and monitoring systems require continuous human attention to identify and track objects across multiple frames. This leads to reduced efficiency, increased workload, and missed observations.

SmartTrack aims to automate this process by detecting and tracking objects in real time, thereby reducing the need for continuous manual monitoring and improving decision-making through real-time insights.

## Objectives

The primary objectives of the project are:

- To develop a real-time object detection and tracking system.
- To integrate a deep learning-based detector with a multi-object tracker.
- To detect and classify common objects from live or recorded video input.
- To maintain persistent object identities across frames.
- To present the results using bounding boxes, class labels, tracking IDs, and FPS information.
- To provide a user-friendly dashboard for monitoring and analysis.
- To make the system adaptable for further enhancement in surveillance and analytics applications.

## Scope of the Project

SmartTrack can be used in multiple domains, including:

- traffic monitoring and vehicle tracking
- pedestrian detection and movement analysis
- surveillance systems
- smart campus monitoring
- retail and crowd analysis
- general-purpose object detection systems

## Key Features

- Real-time video input from webcam or uploaded video files
- Object detection using YOLOv8
- Multi-object tracking with persistent IDs
- Bounding boxes and labels for each tracked object
- Class filtering to monitor selected categories
- FPS monitoring for performance evaluation
- Dashboard-based interface for seamless interaction
- Analytics and reporting support for session results
- Compatibility with modern Python-based computer vision workflows

## Technology Stack

| Component | Technology |
| --- | --- |
| Programming Language | Python |
| Computer Vision | OpenCV |
| Object Detection | YOLOv8 |
| Deep Learning Framework | PyTorch |
| Multi-Object Tracking | DeepSORT |
| Visualization | OpenCV / Streamlit |
| Data Handling | NumPy, Pandas |
| Frontend / Dashboard | Streamlit |

## System Architecture

The workflow of SmartTrack is divided into the following stages:

1. Video input acquisition from webcam or uploaded recording.
2. Frame preprocessing for consistent analysis.
3. Object detection using a YOLO-based model.
4. Filtering of detections based on confidence threshold and selected classes.
5. Tracking objects across multiple frames using DeepSORT.
6. Drawing bounding boxes, labels, and IDs over the current frame.
7. Displaying results in real time and storing relevant analytics.

### High-Level Workflow

Video Input -> Frame Processing -> Object Detection -> Filtering -> Tracking -> Annotation -> Output Dashboard

## Repository Structure

```text
smarttrack/
├── app.py                 # Main dashboard / web interface
├── main.py                # Real-time detection and tracking runner
├── detector.py            # YOLOv8-based detection logic
├── tracker.py             # DeepSORT tracking logic
├── utils.py               # Visualization and helper functions
├── data/                  # Project data and output directory
├── .streamlit/            # Streamlit configuration files
├── home_bg.png            # Dashboard background image
├── tracking_bg.png        # Tracking UI background image
├── yolov8n.pt             # YOLOv8 model file
├── README.md              # Project documentation
├── .yolov8s.pt...part     # Temporary model-download artifact
└── ...
```

## Software Requirements

The project is designed for the following environment:

- Python 3.9 or later
- Windows, Linux, or macOS
- Webcam or recorded video input
- GPU support recommended for better performance, though CPU execution is supported

## Installation

1. Clone the repository:

```bash
git clone https://github.com/bwtf1/smarttrack.git
cd smarttrack
```

2. Create a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install ultralytics opencv-python numpy pandas deep-sort-realtime streamlit
```

4. Ensure that the YOLO model weight file (`yolov8n.pt`) is available locally or can be downloaded automatically.

## Running the Project

### 1. Run the real-time detection script

```bash
python main.py
```

This opens the default webcam, performs detection and tracking, and displays results in a live video window.

### 2. Run the Streamlit dashboard

```bash
streamlit run app.py
```

This launches the interactive monitoring dashboard where the user can:

- select the input source
- choose detection confidence
- filter object classes
- monitor live tracking results
- analyze sessions and output statistics

## Default Application Credentials

The application includes demo login accounts for testing:

- Username: `admin`  Password: `Admin@123`
- Username: `user`   Password: `User@123`

## Working Methodology

SmartTrack follows a standard deep learning-based computer vision pipeline:

1. Input frames are captured either from a webcam or a video source.
2. The YOLOv8 model detects objects in each frame.
3. Confidence thresholding removes low-quality predictions.
4. Detected objects are filtered by the selected class list.
5. DeepSORT associates objects across frames and assigns tracking IDs.
6. Bounding boxes and labels are drawn on each frame.
7. The frame is displayed in real time along with object counts and FPS.
8. Results can be explored through the dashboard and analytics panels.

## Functional Features

### Object Detection

The detector identifies common objects from the COCO-trained model such as:

- person
- car
- bicycle
- motorcycle
- bus
- truck
- and other supported categories

### Object Tracking

Tracking is performed by assigning each detected object a persistent ID and tracking it across subsequent frames, even when motion changes or there are temporary occlusions.

### Visualization

Detected objects are annotated with:

- bounding boxes
- object labels
- tracking IDs
- FPS overlays

### Analytics and Monitoring

The project includes a monitoring dashboard with analytics such as:

- session summaries
- object counts
- FPS measurements
- class-based insights
- saved tracking history

## Screenshots

The project UI includes a modern dark-themed dashboard and live monitoring interface. The screenshots below demonstrate the SmartTrack workflow and the real-time tracking experience.

Figure 1: SmartTrack dashboard overview

Figure 2: Real-time live monitoring interface with object detection and tracking overlays

> Add the final screenshots in the project documentation folder or update this section with the exact image paths once screenshots are captured in the repository.

## Use Cases

SmartTrack is suitable for:

- real-time object tracking in urban traffic scenarios
- student project demonstrations in computer vision
- intelligent surveillance prototypes
- campus and public-area monitoring
- object movement analysis in live streams

## Challenges and Future Enhancements

Although the current implementation provides a robust foundation, several enhancements can further improve the system:

- improved tracking in crowded or densely populated environments
- real-time counting and line-crossing detection
- recording of annotated output videos
- export of CSV logs and analytics reports
- deployment on embedded or edge devices
- support for additional object classes and domain-specific models

## License

No explicit license has been provided for this project. The repository is currently unlicensed.

## Acknowledgement

We express our sincere gratitude to our faculty guide, Anjali Shrivastav, for her guidance, encouragement, and support throughout this project. We also thank ITS Engineering College for providing the platform and academic environment necessary for the successful completion of this work.

## Conclusion

SmartTrack demonstrates the practical application of computer vision and deep learning in solving real-world object detection and tracking challenges. By combining YOLOv8 with a tracking framework, the project provides a reliable and scalable foundation for intelligent monitoring systems. It is a strong academic project that can be extended to advanced surveillance, smart city, and analytics-based applications.

---

Developed by: SmartTrack Team  
Institution: ITS Engineering College  
Faculty Guide: Anjali Shrivastav
