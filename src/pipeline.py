"""영상 객체 탐지·추적 파이프라인.

Gradio UI(app.py)와 분리해 두어 CLI/노트북/서버 어디서든 재사용할 수 있다.
알고리즘은 직접 학습하지 않고 off-the-shelf 모델(Ultralytics YOLO + ByteTrack/BoT-SORT)을 사용한다.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

# UI 라벨 -> Ultralytics 가중치 이름
AVAILABLE_MODELS: Dict[str, str] = {
    "YOLO11n": "yolo11n.pt",
    "YOLO11s": "yolo11s.pt",
    "YOLOv8n": "yolov8n.pt",
    "YOLOv8s": "yolov8s.pt",
}

AVAILABLE_TRACKERS: Dict[str, str] = {
    "ByteTrack": "bytetrack.yaml",
    "BoT-SORT": "botsort.yaml",
}

# 열화상 입력 대응. COCO 사전학습 가중치는 가시광 이미지의 색 분포를 학습했으므로,
# 온도를 색으로 칠한 false-color 영상은 학습 분포에서 크게 벗어난다. 팔레트를 걷어내
# 휘도만 남기면 탐지가 눈에 띄게 살아난다(README의 측정값 참고).
PREPROCESSORS: Dict[str, str] = {
    "없음 (가시광)": "none",
    "그레이스케일 (열화상)": "gray",
    "그레이스케일 + CLAHE (열화상)": "clahe",
}

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_preprocess(frame: np.ndarray, mode: str) -> np.ndarray:
    """추론 전 입력 변환. 결과 영상도 이 프레임 위에 그린다(모델이 본 것을 그대로 보여주기 위해)."""
    if mode == "none":
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mode == "clahe":
        gray = _CLAHE.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

_MODEL_CACHE: Dict[str, "object"] = {}


@dataclass
class TrackingStats:
    """한 번의 추론 실행에 대한 요약 통계."""

    model_name: str
    tracker_name: str
    imgsz: int
    source_fps: float
    frame_count: int
    processed_frames: int
    elapsed_sec: float
    total_infer_ms: float
    unique_tracks: Dict[str, int] = field(default_factory=dict)
    peak_simultaneous: Dict[str, int] = field(default_factory=dict)

    @property
    def avg_infer_ms(self) -> float:
        return self.total_infer_ms / self.processed_frames if self.processed_frames else 0.0

    @property
    def infer_fps(self) -> float:
        """모델 추론만 기준으로 한 FPS (디코딩·인코딩 제외)."""
        return 1000.0 / self.avg_infer_ms if self.avg_infer_ms else 0.0

    @property
    def pipeline_fps(self) -> float:
        """디코딩 + 추론 + 인코딩까지 포함한 실제 처리 FPS."""
        return self.processed_frames / self.elapsed_sec if self.elapsed_sec else 0.0

    @property
    def total_objects(self) -> int:
        return sum(self.unique_tracks.values())

    def to_table(self) -> List[List[object]]:
        """클래스별 카운트 표 (Gradio Dataframe 용)."""
        rows = [
            [cls, count, self.peak_simultaneous.get(cls, 0)]
            for cls, count in sorted(self.unique_tracks.items(), key=lambda kv: -kv[1])
        ]
        return rows or [["(탐지된 객체 없음)", 0, 0]]

    def to_markdown(self) -> str:
        return (
            f"### 처리 결과\n"
            f"- **모델 / 추적기**: `{self.model_name}` + `{self.tracker_name}` (imgsz={self.imgsz})\n"
            f"- **처리 프레임**: {self.processed_frames} / {self.frame_count} "
            f"(원본 {self.source_fps:.1f} FPS)\n"
            f"- **추론 속도**: {self.avg_infer_ms:.1f} ms/frame → **{self.infer_fps:.1f} FPS**\n"
            f"- **전체 파이프라인**: {self.elapsed_sec:.1f} s → **{self.pipeline_fps:.1f} FPS** "
            f"(디코딩·인코딩 포함)\n"
            f"- **누적 추적 객체 수(고유 ID)**: {self.total_objects}\n"
        )


def physical_cores() -> int:
    """추론 스레드 수로 쓸 물리 코어 수. 얻지 못하면 논리 코어의 절반."""
    import os

    try:
        import psutil

        count = psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001 - psutil이 없거나 값을 못 얻는 환경
        count = None
    return count or max(1, (os.cpu_count() or 2) // 2)


def load_model(weights: str):
    """가중치를 캐싱해 재실행 시 로딩 비용을 없앤다."""
    if weights not in _MODEL_CACHE:
        import torch
        from ultralytics import YOLO  # 지연 임포트 (앱 기동 속도 확보)

        # ultralytics는 import 시점에 OMP_NUM_THREADS=1을 설정해 PyTorch를 단일 스레드로
        # 묶는다. import 이후에 스레드 수를 다시 지정하지 않으면 코어를 하나만 쓴다.
        torch.set_num_threads(physical_cores())
        _MODEL_CACHE[weights] = YOLO(weights)
    return _MODEL_CACHE[weights]


def _fourcc(code: str) -> int:
    """OpenCV 4/5 양쪽에서 동작하는 FourCC 헬퍼."""
    fn = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc
    return int(fn(*code))


def _find_ffmpeg() -> Optional[str]:
    """시스템 ffmpeg → imageio-ffmpeg 번들 바이너리 순으로 찾는다."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - 없으면 그냥 재인코딩을 건너뛴다
        return None


def _open_writer(path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    """중간 산출물을 기록할 writer.

    최종 H.264 변환은 ffmpeg가 담당하므로, 여기서는 어디서나 열리는 mp4v를 먼저 쓴다.
    ffmpeg가 아예 없을 때만 브라우저 재생을 위해 avc1을 우선 시도한다.
    """
    codecs = ("mp4v", "avc1") if _find_ffmpeg() else ("avc1", "mp4v")
    for fourcc in codecs:
        writer = cv2.VideoWriter(str(path), _fourcc(fourcc), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("결과 영상을 기록할 코덱을 찾지 못했습니다.")


def _to_browser_mp4(src: Path) -> Path:
    """가능하면 H.264(yuv420p)로 재인코딩해 브라우저에서 바로 재생되게 한다."""
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        return src
    dst = src.with_name(src.stem + "_h264.mp4")
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=600)
    except (subprocess.SubprocessError, OSError):
        return src
    return dst if dst.exists() and dst.stat().st_size > 0 else src


def run_tracking(
    video_path: str,
    model_label: str = "YOLO11n",
    tracker_label: str = "ByteTrack",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    max_seconds: float = 20.0,
    preprocess_label: str = "없음 (가시광)",
    line_width: Optional[int] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Tuple[str, TrackingStats]:
    """영상 한 편을 탐지·추적해 오버레이 영상과 통계를 반환한다.

    Args:
        video_path: 입력 영상 경로.
        model_label / tracker_label: UI 드롭다운 라벨 (AVAILABLE_* 키).
        conf, iou: 탐지 신뢰도 / NMS IoU 임계값.
        imgsz: 추론 입력 해상도. 작을수록 빠르고 작은 객체에 약해진다.
        max_seconds: 무료 CPU 환경 보호용 처리 길이 상한(초). 0 이하면 전체 처리.
        preprocess_label: 추론 전 입력 변환 (PREPROCESSORS 키). 열화상 입력에 사용.
        line_width: 오버레이 선 두께. None이면 해상도에 맞춰 자동. 작게 주면 라벨도 작아진다.
        progress_cb: (0~1 진행률, 메시지)를 받는 콜백. Gradio 진행바 연결용.

    Returns:
        (결과 영상 경로, TrackingStats)
    """
    if not video_path:
        raise ValueError("입력 영상이 없습니다. 영상을 업로드하거나 샘플을 선택하세요.")

    weights = AVAILABLE_MODELS.get(model_label, "yolo11n.pt")
    tracker_cfg = AVAILABLE_TRACKERS.get(tracker_label, "bytetrack.yaml")
    preprocess_mode = PREPROCESSORS.get(preprocess_label, "none")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_budget = frame_count
    if max_seconds and max_seconds > 0:
        frame_budget = min(frame_count or 10**9, int(src_fps * max_seconds))

    model = load_model(weights)

    out_dir = Path(tempfile.mkdtemp(prefix="tracking_"))
    raw_path = out_dir / "result.mp4"
    writer = _open_writer(raw_path, src_fps, (width, height))

    unique_tracks: Dict[str, set] = defaultdict(set)
    peak_simultaneous: Dict[str, int] = defaultdict(int)
    total_infer_ms = 0.0
    processed = 0
    started = time.perf_counter()

    try:
        while processed < frame_budget:
            ok, frame = cap.read()
            if not ok:
                break
            frame = apply_preprocess(frame, preprocess_mode)

            t0 = time.perf_counter()
            # 첫 프레임은 persist=False로 호출해 이전 실행의 추적 상태를 초기화한다.
            results = model.track(
                frame,
                persist=processed > 0,
                tracker=tracker_cfg,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False,
            )
            total_infer_ms += (time.perf_counter() - t0) * 1000.0

            result = results[0]
            writer.write(result.plot(line_width=line_width))

            boxes = result.boxes
            frame_counts: Dict[str, int] = defaultdict(int)
            if boxes is not None and boxes.id is not None:
                names = result.names
                for cls_id, track_id in zip(
                    boxes.cls.int().tolist(), boxes.id.int().tolist()
                ):
                    label = names[int(cls_id)]
                    unique_tracks[label].add(track_id)
                    frame_counts[label] += 1
            for label, count in frame_counts.items():
                peak_simultaneous[label] = max(peak_simultaneous[label], count)

            processed += 1
            if progress_cb and processed % 5 == 0:
                progress_cb(processed / max(frame_budget, 1), f"{processed}/{frame_budget} 프레임")
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - started
    if processed == 0:
        raise ValueError("영상에서 프레임을 읽지 못했습니다. 다른 파일을 시도해 주세요.")

    if progress_cb:
        progress_cb(1.0, "영상 인코딩 중…")
    out_path = _to_browser_mp4(raw_path)

    stats = TrackingStats(
        model_name=weights,
        tracker_name=tracker_cfg.replace(".yaml", ""),
        imgsz=imgsz,
        source_fps=src_fps,
        frame_count=frame_count,
        processed_frames=processed,
        elapsed_sec=elapsed,
        total_infer_ms=total_infer_ms,
        unique_tracks={k: len(v) for k, v in unique_tracks.items()},
        peak_simultaneous=dict(peak_simultaneous),
    )
    return str(out_path), stats


if __name__ == "__main__":  # 간단한 CLI: python -m src.pipeline <video>
    import argparse

    parser = argparse.ArgumentParser(description="YOLO 탐지·추적 파이프라인")
    parser.add_argument("video")
    parser.add_argument("--model", default="YOLO11n", choices=list(AVAILABLE_MODELS))
    parser.add_argument("--tracker", default="ByteTrack", choices=list(AVAILABLE_TRACKERS))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--preprocess", default="없음 (가시광)", choices=list(PREPROCESSORS))
    args = parser.parse_args()

    path, stats = run_tracking(
        args.video,
        model_label=args.model,
        tracker_label=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_seconds=args.max_seconds,
        preprocess_label=args.preprocess,
        progress_cb=lambda p, msg: print(f"\r{p*100:5.1f}% {msg}", end=""),
    )
    print("\n" + stats.to_markdown())
    print(f"결과 영상: {path}")
