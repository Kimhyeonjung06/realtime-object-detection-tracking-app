"""데모용 샘플 영상 내려받기.

사용법:
    python scripts/fetch_samples.py

레포에 큰 영상 파일을 커밋하지 않기 위해, 필요할 때 공개 URL에서 받아 samples/ 에 저장한다.
"""

import sys
import urllib.request
from pathlib import Path

SAMPLES = {
    "people-walking.mp4": "https://media.roboflow.com/supervision/video-examples/people-walking.mp4",
    "vehicles.mp4": "https://media.roboflow.com/supervision/video-examples/vehicles.mp4",
}

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
            urllib.request.urlretrieve(url, target)
        except Exception as exc:  # noqa: BLE001 - 네트워크 오류는 이름만 보고하면 충분
            print(f"[fail] {name}: {exc}")
            target.unlink(missing_ok=True)
            failed.append(name)
            continue
        print(f"[ok  ] {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
