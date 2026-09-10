"""실시간 객체 탐지·추적 웹앱 (Gradio).

영상을 업로드하면 YOLO + ByteTrack으로 객체를 탐지·추적해
바운딩박스/트랙 ID 오버레이 영상과 처리 통계를 반환한다.
"""

from pathlib import Path

import gradio as gr
import pandas as pd

from src.benchmark import results_to_markdown, results_to_table, run_benchmark
from src.pipeline import AVAILABLE_MODELS, AVAILABLE_TRACKERS, run_tracking

SAMPLES_DIR = Path(__file__).parent / "samples"

DESCRIPTION = """
# 실시간 객체 탐지·추적 웹앱

영상을 올리면 **YOLO(탐지) + ByteTrack/BoT-SORT(추적)** 파이프라인이 프레임마다 객체를 찾고
같은 객체에 일관된 **트랙 ID**를 붙여 따라갑니다. 결과는 오버레이 영상과 처리 통계로 나옵니다.

- 모델은 직접 학습하지 않고 **off-the-shelf 사전학습 가중치**(COCO 80클래스)를 사용합니다.
- 표시되는 FPS·지연시간은 **실제 측정값**이며, 실행 환경(CPU/GPU)에 따라 달라집니다.
"""

TIPS = """
**파라미터 가이드**
- `Confidence` 높이면 오탐 감소·미검출 증가, `IoU` 낮추면 겹친 박스 억제 강화
- `추론 해상도` 낮추면 속도 향상, 작은 객체 성능 저하 (속도–정확도 트레이드오프)
- `ByteTrack`은 가볍고 빠름, `BoT-SORT`는 외형 정보까지 써서 가림(occlusion)에 강한 편
- 무료 CPU 환경에서는 **처리 길이 상한**을 짧게 두는 것이 좋습니다.
- 모델 뒤 `n`은 nano(가장 빠름), `s`는 small(정확도 우선)
"""


def analyze(video, model_label, tracker_label, conf, iou, imgsz, max_seconds,
            progress=gr.Progress()):
    if not video:
        raise gr.Error("먼저 영상을 업로드하거나 아래 샘플을 선택하세요.")

    progress(0.0, desc="모델 로딩 중…")
    try:
        out_path, stats = run_tracking(
            video,
            model_label=model_label,
            tracker_label=tracker_label,
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            max_seconds=float(max_seconds),
            progress_cb=lambda p, msg: progress(p, desc=f"추론 중… {msg}"),
        )
    except ValueError as exc:
        raise gr.Error(str(exc))

    return out_path, stats.to_markdown(), stats.to_table()


BENCH_INTRO = """
### 배포 런타임 비교

같은 가중치를 **PyTorch / ONNX Runtime FP32 / ONNX Runtime INT8** 세 형태로 실행해
프레임당 지연시간, 모델 크기, 탐지 품질 손실을 실측합니다. 온디바이스 환경에서는
정확도보다 "이 지연시간과 메모리 안에 들어가는가"가 먼저 걸리기 때문입니다.

- 지연시간은 **전·후처리를 제외한 forward 구간**만, **런타임마다 별도 프로세스**에서 측정합니다.
- INT8은 보정 데이터로 범위를 잡는 **정적 양자화**이며, 영상의 프레임에서 보정 샘플을 뽑습니다.
- 탐지 품질은 라벨이 없으므로 mAP 대신 **PyTorch 출력 대비 재현율**(IoU 0.5, 동일 클래스)로 봅니다.
"""

BENCH_NOTE = """
처음 실행하면 ONNX export와 양자화 보정을 거치느라 몇 분 걸립니다. 변환된 모델은
`models/`에 캐시되어 다음 실행부터는 측정만 수행합니다. YOLO11 계열은 attention 블록
때문에 ONNX Runtime 정적 양자화가 실패하며, 그 경우 INT8 행 대신 사유가 표시됩니다.
"""


def benchmark(video, model_label, imgsz, frames, progress=gr.Progress()):
    if not video:
        raise gr.Error("먼저 영상을 업로드하거나 아래 샘플을 선택하세요.")

    try:
        results, meta = run_benchmark(
            video,
            model_label=model_label,
            imgsz=int(imgsz),
            frame_count=int(frames),
            progress_cb=lambda p, msg: progress(p, desc=msg),
        )
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc))

    table = results_to_table(results)
    latency = pd.DataFrame({"런타임": [r.runtime for r in results],
                            "지연시간(ms)": [round(r.latency_ms, 1) for r in results]})
    size = pd.DataFrame({"런타임": [r.runtime for r in results],
                         "모델 크기(MB)": [round(r.model_mb, 1) for r in results]})
    return results_to_markdown(results, meta), table, latency, size


def build_demo() -> gr.Blocks:
    sample_videos = sorted(str(p) for p in SAMPLES_DIR.glob("*.mp4"))

    with gr.Blocks(title="실시간 객체 탐지·추적 웹앱", theme=gr.themes.Soft()) as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Tab("탐지·추적"):
            _build_tracking_tab(sample_videos)

        with gr.Tab("런타임 벤치마크"):
            _build_benchmark_tab(sample_videos)

    return demo


def _build_benchmark_tab(sample_videos):
    gr.Markdown(BENCH_INTRO)
    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="측정에 쓸 영상", sources=["upload"])
            model_dd = gr.Dropdown(choices=list(AVAILABLE_MODELS), value="YOLOv8n",
                                   label="모델")
            imgsz_dd = gr.Dropdown(choices=[320, 480, 640], value=640, label="추론 해상도")
            frames_sl = gr.Slider(20, 100, value=40, step=10, label="측정 프레임 수")
            run_btn = gr.Button("벤치마크 실행", variant="primary")
            gr.Markdown(BENCH_NOTE)
        with gr.Column(scale=1):
            summary_md = gr.Markdown()
            table = gr.Dataframe(
                headers=["런타임", "지연 중앙값(ms)", "지연 p95(ms)", "FPS",
                         "모델 크기(MB)", "탐지 수", "FP32 대비 일치율"],
                label="측정 결과",
                wrap=True,
            )
            latency_plot = gr.BarPlot(x="런타임", y="지연시간(ms)",
                                      title="프레임당 추론 지연 (낮을수록 좋음)",
                                      vertical=False, height=220)
            size_plot = gr.BarPlot(x="런타임", y="모델 크기(MB)",
                                   title="모델 파일 크기 (낮을수록 좋음)",
                                   vertical=False, height=220)

    if sample_videos:
        gr.Examples(examples=[[v] for v in sample_videos], inputs=[video_in],
                    label="샘플 영상")

    run_btn.click(fn=benchmark,
                  inputs=[video_in, model_dd, imgsz_dd, frames_sl],
                  outputs=[summary_md, table, latency_plot, size_plot])


def _build_tracking_tab(sample_videos):
    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="입력 영상", sources=["upload", "webcam"])
            model_dd = gr.Dropdown(
                choices=list(AVAILABLE_MODELS),
                value=list(AVAILABLE_MODELS)[0],
                label="탐지 모델",
            )
            tracker_dd = gr.Dropdown(
                choices=list(AVAILABLE_TRACKERS),
                value=list(AVAILABLE_TRACKERS)[0],
                label="추적 알고리즘",
            )
            with gr.Accordion("고급 설정", open=False):
                conf_sl = gr.Slider(0.05, 0.95, value=0.25, step=0.05,
                                    label="Confidence threshold")
                iou_sl = gr.Slider(0.1, 0.95, value=0.45, step=0.05,
                                   label="IoU threshold (NMS)")
                imgsz_dd = gr.Dropdown(choices=[320, 480, 640, 960], value=640,
                                       label="추론 해상도 (imgsz)")
                sec_sl = gr.Slider(5, 60, value=20, step=5,
                                   label="처리 길이 상한 (초)")
            run_btn = gr.Button("탐지·추적 실행", variant="primary")

        with gr.Column(scale=1):
            video_out = gr.Video(label="결과 영상 (박스 + 트랙 ID)")
            summary_md = gr.Markdown()
            stats_df = gr.Dataframe(
                headers=["클래스", "누적 추적 수(고유 ID)", "동시 최대 탐지 수"],
                label="클래스별 통계",
                wrap=True,
            )

    gr.Markdown(TIPS)

    if sample_videos:
        gr.Examples(
            examples=[[v] for v in sample_videos],
            inputs=[video_in],
            label="샘플 영상",
        )

    run_btn.click(
        fn=analyze,
        inputs=[video_in, model_dd, tracker_dd, conf_sl, iou_sl, imgsz_dd, sec_sl],
        outputs=[video_out, summary_md, stats_df],
    )


if __name__ == "__main__":
    build_demo().queue().launch()
