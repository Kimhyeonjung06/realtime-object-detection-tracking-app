"""데모용 샘플 영상 내려받기.

사용법:
    python scripts/fetch_samples.py

레포에 큰 영상 파일을 커밋하지 않기 위해, 필요할 때 공개 URL에서 받아 samples/ 에 저장한다.
출처와 라이선스는 samples/SOURCES.md 참고.
"""

import sys
import urllib.request
from pathlib import Path

SAMPLES = {
    "people-walking.mp4": "https://media.roboflow.com/supervision/video-examples/people-walking.mp4",
    "vehicles.mp4": "https://media.roboflow.com/supervision/video-examples/vehicles.mp4",
    # 열화상 원본 (CC BY-SA 4.0, C.Suthorn). 레포에는 10초 발췌본만 커밋되어 있다.
    "thermal-full.webm": "https://upload.wikimedia.org/wikipedia/commons/2/29/"
                         "Thermal_image_video_from_Germany_in_November_2024_03.webm",
}

# Wikimedia는 기본 User-Agent를 차단한다.
USER_AGENT = "realtime-object-detection-tracking-app/1.0 (portfolio project)"

DEST = Path(__file__).resolve().parent.parent / "samples"


def main() -> int:
    DEST.mkdir(exist_ok=True)
    failed = []
    for name, url in SAMPLES.items():
        target = DEST / name
        if target.exists():
            print(f"[skip] {name} (이미 존재)")
            continue
        print(f"[get ] {name} <- {url}")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request) as response, open(target, "wb") as out:
                out.write(response.read())
        except Exception as exc:  # noqa: BLE001 - 네트워크 오류는 이름만 보고하면 충분
            print(f"[fail] {name}: {exc}")
            target.unlink(missing_ok=True)
            failed.append(name)
            continue
        print(f"[ok  ] {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
