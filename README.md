---
title: Realtime Object Detection Tracking
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Object detection and tracking web app built on YOLO and ByteTrack
---

# Real-time Object Detection and Tracking

A web application that detects objects in a video and tracks them across frames. Upload a clip,
and the app returns an annotated video with bounding boxes and persistent track IDs, along with
per-class counts and measured throughput.

Detection uses off-the-shelf pretrained YOLO weights; tracking uses ByteTrack or BoT-SORT.
The point of the project is not a new model but a working, deployed application around one.

## Live demo

https://huggingface.co/spaces/Kimhyeonjung06/realtime-object-detection-tracking

<!-- TODO: verify this URL once the Space is created -->

## Screenshot

![Application screenshot](docs/screenshot.png)

<!-- TODO: save a screenshot of the running app to docs/screenshot.png -->

## Features

- Detection plus tracking: objects keep a consistent ID across frames instead of being re-detected independently.
- Model selection: YOLO11n, YOLO11s, YOLOv8n, YOLOv8s (COCO, 80 classes).
- Tracker selection: ByteTrack (faster) or BoT-SORT (uses appearance cues, more robust under occlusion).
- Adjustable confidence, NMS IoU, and inference resolution, so the speed/accuracy trade-off is visible in the UI.
- Measured performance: milliseconds per frame, inference FPS, and end-to-end pipeline FPS including decode and encode.
- Per-class statistics separating *unique tracks* from *peak simultaneous detections*.
- Input from file upload, webcam, or bundled sample clips.

## How it works

```
video -> OpenCV frame decode
      -> YOLO detection (conf / iou / imgsz)
      -> ByteTrack or BoT-SORT association -> track IDs
      -> overlay rendering + H.264 re-encode
      -> annotated video, throughput metrics, per-class counts
```

Three details worth noting:

- **Tracker state resets per run.** The first frame of each run calls the tracker with `persist=False`,
  so IDs from a previous video never leak into the next one.
- **Counts use unique track IDs, not per-frame detections.** A person visible for 100 frames counts once.
  Peak simultaneous detections are reported separately, since the two numbers answer different questions.
- **Output is re-encoded to H.264 (yuv420p).** OpenCV's default mp4v output is not playable in most
  browsers. If a system `ffmpeg` is unavailable, the binary bundled with `imageio-ffmpeg` is used.

## Running locally

```bash
git clone https://github.com/Kimhyeonjung06/realtime-object-detection-tracking-app.git
cd realtime-object-detection-tracking-app

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/fetch_samples.py   # optional, downloads sample clips
python app.py                     # http://127.0.0.1:7860
```

Model weights (`yolo11n.pt` and friends) download automatically on first use.
Python 3.10 or newer is recommended to match the Hugging Face Spaces runtime; 3.9 also works
through the version markers in `requirements.txt`.

The pipeline runs standalone as well:

```bash
python -m src.pipeline samples/people-walking.mp4 --model YOLO11n --imgsz 480
```

## Tech stack

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-0B5FFF)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?logo=gradio&logoColor=white)

| Area | Technology |
| --- | --- |
| Detection | Ultralytics YOLO11 / YOLOv8, COCO-pretrained |
| Tracking | ByteTrack, BoT-SORT |
| Video | OpenCV, FFmpeg |
| Interface | Gradio Blocks |
| Deployment | Hugging Face Spaces |

## Project layout

```
app.py                    Gradio interface (entry point)
src/pipeline.py           Detection and tracking pipeline, UI-independent, also runnable as a CLI
scripts/fetch_samples.py  Sample video downloader
samples/                  Demo clips
apt.txt                   System packages for Hugging Face Spaces (ffmpeg, libgl1)
requirements.txt
```

The pipeline is kept separate from the interface so the same code can back a web app, a CLI, or a
batch job without modification.

## Notes

- Reported FPS is measured on whatever hardware the app runs on. A free CPU Space produces single-digit
  numbers; a GPU is considerably faster. No figures here are estimated or scaled.
- The models are public pretrained checkpoints, not trained for this project.
- Processing is capped at 20 seconds of video by default to keep free CPU hosting responsive.

## License

MIT
