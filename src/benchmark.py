"""배포 런타임 비교 벤치마크: PyTorch vs ONNX Runtime FP32 vs ONNX Runtime INT8.

온디바이스/임베디드 환경에서는 "정확도가 얼마인가"보다 "이 지연시간과 메모리 안에
들어가는가"가 먼저 걸린다. 이 모듈은 같은 가중치를 세 가지 형태로 실행해
프레임당 지연시간, 모델 크기, 그리고 FP32 대비 탐지 일치율을 실측한다.

INT8 양자화는 정적 양자화(static quantization)를 쓴다. 동적 양자화는 Conv 연산을
양자화하지 않아 CNN에서는 속도 이득이 거의 없기 때문이다. 보정(calibration)
데이터는 벤치마크에 쓰는 영상의 프레임에서 뽑는다.
"""

from __future__ import annotations

import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .pipeline import AVAILABLE_MODELS

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CALIBRATION_FRAMES = 32

_BACKEND_CACHE: Dict[str, object] = {}


@dataclass
class RuntimeResult:
    """한 런타임에 대한 측정 결과."""

    runtime: str
    latency_ms: float          # 프레임당 중앙값
    latency_p95_ms: float
    model_mb: float
    detections: int            # 측정 구간의 총 탐지 수
    agreement: Optional[float] # FP32 기준 대비 탐지 일치율 (기준 런타임은 None)

    @property
    def fps(self) -> float:
        return 1000.0 / self.latency_ms if self.latency_ms else 0.0


# --------------------------------------------------------------------------- #
# 프레임 준비
# --------------------------------------------------------------------------- #

def sample_frames(video_path: str, count: int, stride: int = 1) -> List[np.ndarray]:
    """영상에서 프레임을 균등하게 뽑는다."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")
    frames: List[np.ndarray] = []
    idx = 0
    try:
        while len(frames) < count:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                frames.append(frame)
            idx += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError("영상에서 프레임을 읽지 못했습니다.")
    return frames


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    """비율을 유지한 채 정사각형으로 패딩 (YOLO 전처리와 동일한 방식)."""
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def to_input_tensor(image: np.ndarray, size: int) -> np.ndarray:
    """BGR uint8 프레임 -> NCHW float32 [0,1] 텐서."""
    padded = letterbox(image, size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    # transpose 결과는 non-contiguous라 그대로 넘기면 conv 커널이 느린 경로를 탄다.
    return np.ascontiguousarray(chw[None, ...])


# --------------------------------------------------------------------------- #
# 모델 변환
# --------------------------------------------------------------------------- #

def export_onnx(weights: str, imgsz: int) -> Path:
    """Ultralytics 가중치를 ONNX로 export (이미 있으면 재사용)."""
    from ultralytics import YOLO

    MODELS_DIR.mkdir(exist_ok=True)
    target = MODELS_DIR / f"{Path(weights).stem}_{imgsz}.onnx"
    if target.exists():
        return target

    exported = YOLO(weights).export(format="onnx", imgsz=imgsz, opset=13, simplify=True)
    shutil.move(str(exported), target)
    return target


def quantize_int8(onnx_path: Path, calib_frames: Sequence[np.ndarray], imgsz: int) -> Path:
    """정적 양자화로 INT8 모델을 만든다 (이미 있으면 재사용).

    양자화는 ASCII 경로의 임시 디렉터리에서 수행한 뒤 결과만 복사해 온다.
    ONNX의 shape inference가 C++ 계층에서 경로를 바이트 문자열로 다루기 때문에,
    저장소가 한글 등 non-ASCII 경로에 있으면 그대로는 실패한다.
    """
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    target = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
    if target.exists():
        return target

    input_name = _onnx_input_name(onnx_path)

    class FrameReader(CalibrationDataReader):
        """보정용 프레임을 한 장씩 흘려보낸다."""

        def __init__(self) -> None:
            self._it = iter(calib_frames)

        def get_next(self):
            frame = next(self._it, None)
            if frame is None:
                return None
            return {input_name: to_input_tensor(frame, imgsz)}

    with tempfile.TemporaryDirectory(prefix="quant_") as tmp:
        work = Path(tmp)
        staged = work / "model.onnx"
        prepared = work / "model_prep.onnx"
        quantized = work / "model_int8.onnx"

        shutil.copy(onnx_path, staged)
        quant_pre_process(str(staged), str(prepared), skip_symbolic_shape=True)
        quantize_static(
            model_input=str(prepared),
            model_output=str(quantized),
            calibration_data_reader=FrameReader(),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            # Conv만 양자화한다. 그래프 전체를 양자화하면 박스 좌표(0~640)와 클래스 점수(0~1)가
            # 함께 담긴 출력 텐서에 8비트 스케일 하나(실측 2.5)가 걸려, 점수가 전부 0으로
            # 반올림되고 탐지가 0개가 된다.
            op_types_to_quantize=["Conv"],
            calibrate_method=CalibrationMethod.MinMax,
        )
        shutil.copy(quantized, target)
    return target


def _onnx_input_name(path: Path) -> str:
    import onnx

    return onnx.load(str(path)).graph.input[0].name


# --------------------------------------------------------------------------- #
# 측정
# --------------------------------------------------------------------------- #

def _load_backend(source: str):
    """세 런타임 모두 Ultralytics 인터페이스로 감싸 동일 조건에서 비교한다."""
    if source not in _BACKEND_CACHE:
        from ultralytics import YOLO

        _BACKEND_CACHE[source] = YOLO(source, task="detect")
    return _BACKEND_CACHE[source]


def bench_threads() -> int:
    """측정에 쓸 스레드 수: 물리 코어 수로 고정한다.

    런타임마다 기본 스레드 정책이 달라 명시적으로 맞춰야 한다. 이 값을 라이브러리의
    현재 상태(torch.get_num_threads 등)에서 읽어오면 안 된다. ultralytics는 import 시점에
    OMP_NUM_THREADS=1을 프로세스 환경에 써 넣기 때문에, 그 뒤에 읽으면 1이 나온다.
    """
    from .pipeline import physical_cores

    return physical_cores()


def measure_inference(kind: str, source: str, tensors: Sequence[np.ndarray]) -> List[float]:
    """각 런타임을 별도 프로세스에서, 스레드 수를 명시적으로 고정해 측정한다.

    처음 구현은 자식 프로세스에 부모 환경을 그대로 물려줬다. 부모가 ultralytics를
    import하면서 OMP_NUM_THREADS=1이 들어가 있었고, 그 결과 자식의 PyTorch와
    ONNX Runtime이 모두 단일 스레드로 돌아 지연이 2.5~3배 부풀려졌다(실측: ONNX
    Runtime FP32 30ms -> 87ms). 그래서 스레드 수를 환경과 인자 양쪽에서 강제한다.
    """
    import json
    import os
    import subprocess
    import sys

    threads = bench_threads()
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["BENCH_THREADS"] = str(threads)

    with tempfile.TemporaryDirectory(prefix="bench_") as tmp:
        payload = Path(tmp) / "tensors.npy"
        np.save(payload, np.concatenate(list(tensors), axis=0))
        proc = subprocess.run(
            [sys.executable, "-m", "src.benchmark", "--worker", kind, source, str(payload)],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=env,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"{kind} 측정 실패: {proc.stderr.strip()[-400:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _measure_in_process(kind: str, source: str, tensors: Sequence[np.ndarray]) -> List[float]:
    """워커 프로세스에서 실제로 도는 측정 루프 (전·후처리 제외한 forward 구간)."""
    import os

    threads = int(os.environ.get("BENCH_THREADS") or bench_threads())

    if kind == "torch":
        import torch

        from ultralytics import YOLO

        # ultralytics import가 OMP_NUM_THREADS를 다시 1로 덮어쓰므로 import 이후에 고정한다.
        torch.set_num_threads(threads)

        # ONNX export는 conv+bn이 합쳐진 그래프를 내보내므로, PyTorch 쪽도 fuse해야
        # 같은 연산량을 비교하게 된다. fuse를 빼면 PyTorch가 부당하게 느리게 나온다.
        net = YOLO(source).model.float().eval().fuse()
        run = lambda t: net(torch.from_numpy(t))  # noqa: E731
        with torch.inference_mode():
            run(tensors[0])
            latencies = []
            for t in tensors:
                t0 = time.perf_counter()
                run(t)
                latencies.append((time.perf_counter() - t0) * 1000.0)
        return latencies

    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(source, opts, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    sess.run(None, {name: tensors[0]})
    latencies = []
    for t in tensors:
        t0 = time.perf_counter()
        sess.run(None, {name: t})
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def _collect_detections(source: str, frames: Sequence[np.ndarray], imgsz: int, conf: float,
                        iou: float) -> List[np.ndarray]:
    """탐지 품질 비교용 결과 수집 (전·후처리 포함, Ultralytics 경로)."""
    model = _load_backend(source)
    detections: List[np.ndarray] = []
    for frame in frames:
        result = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            detections.append(np.zeros((0, 6), dtype=np.float32))
        else:
            detections.append(
                np.concatenate(
                    [boxes.xyxy.cpu().numpy(),
                     boxes.conf.cpu().numpy()[:, None],
                     boxes.cls.cpu().numpy()[:, None]],
                    axis=1,
                ).astype(np.float32)
            )
    return detections


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """두 박스 집합 간 IoU 행렬."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def detection_agreement(baseline: Sequence[np.ndarray], candidate: Sequence[np.ndarray],
                        iou_thr: float = 0.5) -> float:
    """FP32 기준 탐지 중 몇 %가 같은 클래스·같은 위치로 재현되는지.

    라벨이 없는 영상이므로 mAP은 계산할 수 없다. 대신 FP32 출력을 기준으로 삼아
    "양자화 후 무엇을 잃었는가"를 재현율 형태로 측정한다.
    """
    matched = total = 0
    for ref, cand in zip(baseline, candidate):
        total += len(ref)
        if len(ref) == 0 or len(cand) == 0:
            continue
        ious = _iou_matrix(ref, cand)
        same_class = ref[:, 5][:, None] == cand[None, :, 5]
        matched += int(((ious >= iou_thr) & same_class).any(axis=1).sum())
    return matched / total if total else 0.0


def run_benchmark(
    video_path: str,
    model_label: str = "YOLOv8n",
    imgsz: int = 640,
    frame_count: int = 30,
    conf: float = 0.25,
    iou: float = 0.45,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Tuple[List[RuntimeResult], Dict[str, float]]:
    """세 런타임을 같은 프레임에 대해 실행하고 결과를 모은다."""
    weights = AVAILABLE_MODELS.get(model_label, "yolo11n.pt")

    def step(p: float, msg: str) -> None:
        if progress_cb:
            progress_cb(p, msg)

    step(0.05, "프레임 추출 중")
    frames = sample_frames(video_path, frame_count)
    calib = sample_frames(video_path, CALIBRATION_FRAMES, stride=5)

    step(0.15, "ONNX export 중")
    onnx_path = export_onnx(weights, imgsz)

    step(0.30, "INT8 정적 양자화 중 (보정 데이터 통과)")
    int8_path: Optional[Path] = None
    int8_error = ""
    try:
        int8_path = quantize_int8(onnx_path, calib, imgsz)
    except Exception as exc:  # noqa: BLE001 - 모델 구조에 따라 양자화가 불가능할 수 있다
        int8_error = f"{type(exc).__name__}: {exc}"

    tensors = [to_input_tensor(f, imgsz) for f in frames]

    # 지연 측정을 먼저 끝낸다. 부모 프로세스에서 추론을 돌리고 나면 그 런타임의
    # 스레드가 남아 다음 측정의 코어를 갉아먹기 때문이다.
    step(0.45, "PyTorch 측정 중")
    pt_lat = measure_inference("torch", weights, tensors)

    step(0.60, "ONNX Runtime FP32 측정 중")
    onnx_lat = measure_inference("onnx", str(onnx_path), tensors)

    if int8_path is not None:
        step(0.72, "ONNX Runtime INT8 측정 중")
        int8_lat = measure_inference("onnx", str(int8_path), tensors)

    step(0.82, "탐지 결과 수집 중 (품질 비교)")
    pt_det = _collect_detections(weights, frames, imgsz, conf, iou)
    onnx_det = _collect_detections(str(onnx_path), frames, imgsz, conf, iou)
    if int8_path is not None:
        int8_det = _collect_detections(str(int8_path), frames, imgsz, conf, iou)

    def size_mb(path: Path | str) -> float:
        p = Path(path)
        if not p.exists():  # 가중치는 작업 디렉터리에 자동 다운로드된다
            p = Path.cwd() / Path(path).name
        return p.stat().st_size / 1e6 if p.exists() else 0.0

    results = [
        RuntimeResult("PyTorch (FP32)", statistics.median(pt_lat), _p95(pt_lat),
                      size_mb(weights), sum(len(d) for d in pt_det), None),
        RuntimeResult("ONNX Runtime (FP32)", statistics.median(onnx_lat), _p95(onnx_lat),
                      size_mb(onnx_path), sum(len(d) for d in onnx_det),
                      detection_agreement(pt_det, onnx_det)),
    ]
    if int8_path is not None:
        results.append(
            RuntimeResult("ONNX Runtime (INT8)", statistics.median(int8_lat), _p95(int8_lat),
                          size_mb(int8_path), sum(len(d) for d in int8_det),
                          detection_agreement(pt_det, int8_det))
        )

    baseline = results[0]
    meta: Dict[str, object] = {
        "frames": len(frames),
        "imgsz": imgsz,
        "model": weights,
        "int8_error": int8_error,
        "speedup_onnx": baseline.latency_ms / results[1].latency_ms if results[1].latency_ms else 0.0,
    }
    if int8_path is not None:
        meta["speedup_int8"] = baseline.latency_ms / results[2].latency_ms if results[2].latency_ms else 0.0
        meta["shrink_int8"] = baseline.model_mb / results[2].model_mb if results[2].model_mb else 0.0
    step(1.0, "완료")
    return results, meta


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def results_to_table(results: Sequence[RuntimeResult]) -> List[List[object]]:
    """Gradio Dataframe 용 행 목록."""
    rows = []
    for r in results:
        rows.append([
            r.runtime,
            round(r.latency_ms, 1),
            round(r.latency_p95_ms, 1),
            round(r.fps, 1),
            round(r.model_mb, 1),
            r.detections,
            "기준" if r.agreement is None else f"{r.agreement * 100:.1f}%",
        ])
    return rows


def results_to_markdown(results: Sequence[RuntimeResult], meta: Dict[str, object]) -> str:
    lines = [
        "### 벤치마크 요약",
        f"- **측정 조건**: `{meta['model']}`, {meta['frames']}프레임, "
        f"imgsz={meta['imgsz']}, CPU",
        f"- **ONNX Runtime FP32**: PyTorch 대비 **{meta['speedup_onnx']:.2f}x**",
    ]
    if len(results) > 2:
        int8 = results[2]
        lines += [
            f"- **ONNX Runtime INT8**: PyTorch 대비 **{meta['speedup_int8']:.2f}x**, "
            f"모델 크기 **{meta['shrink_int8']:.1f}배 축소**",
            f"- **INT8 탐지 일치율**: FP32 기준 탐지의 **{int8.agreement * 100:.1f}%** 재현 "
            f"(IoU 0.5, 같은 클래스)",
        ]
    else:
        lines.append(
            f"- **INT8**: 이 모델은 정적 양자화에 실패했습니다 — `{meta['int8_error']}`\n"
            f"  YOLO11 계열의 attention 블록(C2PSA)이 ONNX Runtime 정적 양자화에서 "
            f"지원되지 않습니다. attention이 없는 YOLOv8 계열을 선택하면 측정됩니다."
        )
    lines.append(
        "\n지연시간은 중앙값이며 p95를 함께 표시합니다. 실시간 시스템에서는 평균보다 "
        "꼬리 지연(p95)이 프레임 드롭을 좌우하기 때문입니다."
    )
    return "\n".join(lines)


def _run_worker(argv: Sequence[str]) -> None:
    """`--worker <kind> <source> <tensors.npy>` 로 호출되는 측정 전용 진입점."""
    import json

    kind, source, payload = argv
    stacked = np.load(payload)
    tensors = [stacked[i:i + 1] for i in range(len(stacked))]
    print(json.dumps(_measure_in_process(kind, source, tensors)))


if __name__ == "__main__":
    import argparse
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _run_worker(sys.argv[2:])
        raise SystemExit(0)

    parser = argparse.ArgumentParser(description="런타임 비교 벤치마크")
    parser.add_argument("video")
    parser.add_argument("--model", default="YOLOv8n", choices=list(AVAILABLE_MODELS))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()

    res, meta = run_benchmark(
        args.video, model_label=args.model, imgsz=args.imgsz, frame_count=args.frames,
        progress_cb=lambda p, m: print(f"[{p*100:5.1f}%] {m}"),
    )
    header = ["runtime", "median ms", "p95 ms", "FPS", "MB", "detections", "agreement"]
    print("\n" + " | ".join(header))
    for row in results_to_table(res):
        print(" | ".join(str(c) for c in row))
    print("\n" + results_to_markdown(res, meta))
