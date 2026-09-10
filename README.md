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

## Demo

![Detection and tracking output](docs/demo.gif)

Two seconds of output from the bundled sample clip, produced by YOLO11n with ByteTrack at an
inference size of 640. Every box carries a track ID, and a given person keeps the same ID from
frame to frame rather than being re-detected as someone new. The full run over this clip
registered 52 unique tracks.

<!-- TODO: add docs/screenshot.png, a capture of the Gradio interface -->

## Features

- Detection plus tracking: objects keep a consistent ID across frames instead of being re-detected independently.
- Model selection: YOLO11n, YOLO11s, YOLOv8n, YOLOv8s (COCO, 80 classes).
- Tracker selection: ByteTrack (faster) or BoT-SORT (uses appearance cues, more robust under occlusion).
- Adjustable confidence, NMS IoU, and inference resolution, so the speed/accuracy trade-off is visible in the UI.
- Measured performance: milliseconds per frame, inference FPS, and end-to-end pipeline FPS including decode and encode.
- Per-class statistics separating *unique tracks* from *peak simultaneous detections*.
- Thermal input support: a preprocessing selector that makes an RGB-pretrained model usable on
  infrared footage (measured below).
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

## Thermal (infrared) input

Surveillance and targeting systems rarely see only daylight RGB, so the app carries a preprocessing
selector and a thermal sample clip. The interesting part is how badly a COCO-pretrained,
visible-light model handles infrared, and how much of that gap is closed by a two-line change.

![False-color versus grayscale thermal detection](docs/thermal-comparison.png)

Both panels are the same frame. On the left, two people stand out clearly to a human eye and the
detector finds neither; on the right, the identical frame with the color palette stripped out
yields both.

Across the whole 10-second clip (151 frames, YOLO11n, 640, conf 0.25):

| Preprocessing | Detections | of which person | Most frequent false classes |
| --- | --- | --- | --- |
| As captured (false-color) | 59 | 5 | kite 35, traffic light 16 |
| Grayscale | 132 | 60 | airplane 30, truck 15 |
| Grayscale + CLAHE | 152 | 46 | truck 34, skateboard 34 |

Run through the tracker, that becomes 0, 3 and 4 unique person tracks respectively.

- **The palette, not the sensor, is what breaks the model.** A thermal camera's ironbow mapping
  paints temperature as hue, so a warm body arrives as a saturated orange blob — a color
  distribution nothing in COCO resembles. Discarding the palette and keeping luminance raises
  person detections from 5 to 60, a 12× difference from a color-space conversion.
- **CLAHE is not a free win.** Local contrast equalization finds more objects overall but fewer
  people, and it invents trucks and skateboards out of amplified background texture. It is offered
  as an option rather than a default because the measurement did not support making it one.
- **Preprocessing narrows the domain gap; it does not close it.** Even the best configuration
  reports ~30 phantom aircraft in ten seconds of street footage. A visible-light detector is not
  an infrared detector, and the real fix is training data from the target sensor — this toggle
  makes the size of that gap visible instead of papering over it.

```bash
python -m src.pipeline samples/thermal-street.mp4 --preprocess "그레이스케일 (열화상)"
```

## Deployment runtime benchmark

A second tab runs the same weights through **PyTorch, ONNX Runtime FP32, and ONNX Runtime INT8**
and measures what actually decides whether a model ships to a device: per-frame latency, file
size, and how much detection quality the conversion costs.

Measured on an AMD Ryzen 5 5600 (6 cores, AVX2, no VNNI), YOLOv8n, 640×640, 40 frames:

| Runtime | Median | p95 | Model | Agreement with PyTorch |
| --- | --- | --- | --- | --- |
| PyTorch FP32 | 113 ms | 122 ms | 6.5 MB | baseline |
| ONNX Runtime FP32 | 90 ms (1.26×) | 108 ms | 12.9 MB | 99.1% |
| ONNX Runtime INT8 | 75 ms (1.51×) | 84 ms | 3.6 MB | 93.6% |

Latency is the forward pass only, with pre- and post-processing excluded. There are no labels for
this footage, so instead of reporting an mAP that could not be computed honestly, quality is the
share of PyTorch detections the converted model reproduces at IoU 0.5 with the same class.

Four things this exercise turned up, all of which shaped the implementation:

- **Quantizing the whole graph produces a model that detects nothing.** Static INT8 across every
  op returned zero detections on every frame: the box-decoding arithmetic in the detection head
  does not survive 8-bit. Restricting quantization to `Conv` keeps 93.6% of detections.
- **YOLO11 cannot be statically quantized by ONNX Runtime at all.** Its C2PSA attention block
  fails in the quantizer (`Only an existing tensor can be modified, '.../attn/Softmax_output_0'`),
  and excluding those nodes does not help — the calibrator has already recorded the tensor. The
  benchmark reports this rather than hiding it, and YOLOv8 is the default for this tab. Model
  architecture constrains the deployment path, not just the accuracy number.
- **Measuring several runtimes in one process gives wrong numbers.** Thread pools from an earlier
  runtime stay alive and take cores from the next measurement; PyTorch measured 41 ms alone and
  119 ms after an ONNX Runtime session had been created. Each runtime is now timed in its own
  subprocess, and repeated runs agree within a few percent.
- **INT8 pays off in size unconditionally, in latency only with hardware support.** This CPU has
  AVX2 but no VNNI, so the 1.8× smaller model buys 1.2× over ONNX FP32 rather than the larger
  gains VNNI-capable silicon or an NPU would give.

```bash
python -m src.benchmark samples/people-walking.mp4 --model YOLOv8n --frames 40
```

Exported and quantized models are cached under `models/`, so only the first run pays the
conversion cost.

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
| Deployment | ONNX, ONNX Runtime, INT8 static quantization |
| Video | OpenCV, FFmpeg |
| Interface | Gradio Blocks |
| Deployment | Hugging Face Spaces |

## Project layout

```
app.py                    Gradio interface (entry point)
src/pipeline.py           Detection and tracking pipeline, UI-independent, also runnable as a CLI
src/benchmark.py          ONNX export, INT8 static quantization, runtime measurement
scripts/fetch_samples.py  Sample video downloader
samples/                  Demo clips, including thermal footage (see samples/SOURCES.md)
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
