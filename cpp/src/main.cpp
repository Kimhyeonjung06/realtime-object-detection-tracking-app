// YOLO ONNX 추론 CLI.
//
// 파이썬 런타임 없이 실행 파일 + ONNX 모델 + onnxruntime 공유 라이브러리만으로 동작한다.
// 같은 모델을 파이썬 쪽 벤치마크와 비교하면, 런타임 언어가 아니라 무엇이 실제 지연을
// 만드는지 확인할 수 있다.
//
//   yolo_infer <model.onnx> <image|directory> [옵션]
//     --imgsz N     추론 해상도 (기본 640)
//     --conf F      신뢰도 임계값 (기본 0.25)
//     --iou F       NMS IoU 임계값 (기본 0.45)
//     --threads N   추론 스레드 수 (기본: ONNX Runtime 기본값)
//     --repeat N    이미지마다 N번 반복 측정 (기본 1)
//     --out DIR     탐지 결과를 그려 저장할 디렉터리

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "image_io.hpp"
#include "yolo_detector.hpp"

namespace fs = std::filesystem;

namespace {

struct Args {
    fs::path model;
    fs::path input;
    fs::path out_dir;
    yolo::Options options;
    int repeat = 1;
};

void print_usage() {
    std::cout << "사용법: yolo_infer <model.onnx> <image|directory> "
                 "[--imgsz N] [--conf F] [--iou F] [--threads N] [--repeat N] [--out DIR]\n";
}

// 플래그와 그 값(숫자)은 항상 ASCII이므로 좁은 문자열로 다뤄도 안전하다.
// 경로만 fs::path 그대로 유지한다.
bool parse_args(const std::vector<fs::path>& argv, Args& args) {
    if (argv.size() < 3) return false;
    args.model = argv[1];
    args.input = argv[2];

    for (size_t i = 3; i < argv.size(); ++i) {
        const std::string flag = yolo::display(argv[i]);
        const bool has_value = (i + 1 < argv.size());
        if (flag == "--imgsz" && has_value) {
            args.options.imgsz = std::stoi(yolo::display(argv[++i]));
        } else if (flag == "--conf" && has_value) {
            args.options.conf_threshold = std::stof(yolo::display(argv[++i]));
        } else if (flag == "--iou" && has_value) {
            args.options.iou_threshold = std::stof(yolo::display(argv[++i]));
        } else if (flag == "--threads" && has_value) {
            args.options.threads = std::stoi(yolo::display(argv[++i]));
        } else if (flag == "--repeat" && has_value) {
            args.repeat = std::max(1, std::stoi(yolo::display(argv[++i])));
        } else if (flag == "--out" && has_value) {
            args.out_dir = argv[++i];
        } else {
            std::cerr << "알 수 없는 인자: " << flag << "\n";
            return false;
        }
    }
    return true;
}

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(q * (values.size() - 1) + 0.5);
    return values[std::min(index, values.size() - 1)];
}

}  // namespace

int run(const std::vector<fs::path>& argv) {
    Args args;
    if (!parse_args(argv, args)) {
        print_usage();
        return 1;
    }

    const std::vector<fs::path> files = yolo::collect_images(args.input);
    if (files.empty()) {
        std::cerr << "이미지를 찾지 못했습니다: " << yolo::display(args.input) << "\n";
        return 1;
    }

    if (!args.out_dir.empty()) {
        fs::create_directories(args.out_dir);
    }

    std::cout << "모델    : " << yolo::display(args.model.filename()) << "\n"
              << "입력    : " << files.size() << "장 (imgsz=" << args.options.imgsz
              << ", conf=" << args.options.conf_threshold << ")\n";

    std::unique_ptr<yolo::Detector> detector;
    try {
        detector = std::make_unique<yolo::Detector>(args.model, args.options);
    } catch (const Ort::Exception& error) {
        std::cerr << "모델을 열지 못했습니다: " << error.what() << "\n";
        return 1;
    }

    std::vector<double> latencies;
    std::map<std::string, int> class_counts;
    int total_detections = 0;
    bool warmed_up = false;

    for (const fs::path& file : files) {
        const yolo::Image image = yolo::load_image(file);
        if (!image.valid()) {
            std::cerr << "건너뜀 (읽기 실패): " << yolo::display(file) << "\n";
            continue;
        }

        if (!warmed_up) {
            // 첫 호출에는 메모리 할당과 커널 선택 비용이 섞이므로 측정에서 제외한다.
            detector->detect(image);
            warmed_up = true;
        }

        std::vector<yolo::Detection> detections;
        for (int r = 0; r < args.repeat; ++r) {
            double elapsed = 0.0;
            detections = detector->detect(image, &elapsed);
            latencies.push_back(elapsed);
        }

        total_detections += static_cast<int>(detections.size());
        for (const auto& det : detections) {
            class_counts[detector->class_name(det.class_id)]++;
        }

        if (!args.out_dir.empty()) {
            yolo::Image annotated = image;
            for (const auto& det : detections) {
                yolo::draw_rect(annotated, static_cast<int>(det.x0), static_cast<int>(det.y0),
                                static_cast<int>(det.x1), static_cast<int>(det.y1), 40, 120, 214);
            }
            const fs::path out = args.out_dir / file.filename();
            if (!yolo::save_jpeg(out, annotated)) {
                std::cerr << "저장 실패: " << yolo::display(out) << "\n";
            }
        }
    }

    if (latencies.empty()) {
        std::cerr << "측정된 프레임이 없습니다.\n";
        return 1;
    }

    std::cout << std::fixed << std::setprecision(1)
              << "\n추론 지연 : 중앙값 " << median(latencies) << " ms, p95 "
              << percentile(latencies, 0.95) << " ms  (" << latencies.size() << "회)\n"
              << "처리율    : " << 1000.0 / median(latencies) << " FPS\n"
              << "탐지      : 총 " << total_detections << "개\n";

    for (const auto& [name, count] : class_counts) {
        std::cout << "  " << std::setw(16) << std::left << name << count << "\n";
    }
    return 0;
}

// Windows에서는 wmain으로 받아야 argv가 유니코드로 들어온다. main(char**)의 argv는
// 시스템 코드페이지로 인코딩되어 있어 ASCII 밖의 경로가 전달되면 복원할 수 없다.
#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    return run(std::vector<fs::path>(argv, argv + argc));
}
#else
int main(int argc, char** argv) {
    return run(std::vector<fs::path>(argv, argv + argc));
}
#endif
