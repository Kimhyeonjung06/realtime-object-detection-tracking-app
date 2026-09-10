#include "yolo_detector.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <numeric>
#include <stdexcept>

namespace yolo {
namespace {

const std::vector<std::string>& coco_names() {
    static const std::vector<std::string> names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"};
    return names;
}

float iou(const Detection& a, const Detection& b) {
    const float inter_w = std::max(0.0f, std::min(a.x1, b.x1) - std::max(a.x0, b.x0));
    const float inter_h = std::max(0.0f, std::min(a.y1, b.y1) - std::max(a.y0, b.y0));
    const float inter = inter_w * inter_h;
    const float area_a = (a.x1 - a.x0) * (a.y1 - a.y0);
    const float area_b = (b.x1 - b.x0) * (b.y1 - b.y0);
    const float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

// 클래스별로 독립 적용한다. 서로 다른 클래스의 박스가 겹치는 것은 억제 대상이 아니다.
std::vector<Detection> non_max_suppression(std::vector<Detection> boxes, float threshold) {
    std::sort(boxes.begin(), boxes.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });

    std::vector<Detection> kept;
    std::vector<bool> removed(boxes.size(), false);
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (removed[i]) continue;
        kept.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (removed[j]) continue;
            if (boxes[j].class_id == boxes[i].class_id && iou(boxes[i], boxes[j]) > threshold) {
                removed[j] = true;
            }
        }
    }
    return kept;
}

}  // namespace

Detector::Detector(const std::filesystem::path& model_path, const Options& options)
    : options_(options), env_(ORT_LOGGING_LEVEL_WARNING, "yolo") {
    if (options_.threads > 0) {
        session_options_.SetIntraOpNumThreads(options_.threads);
    }
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // path::value_type은 Windows에서 wchar_t, 그 외에서 char다. ONNX Runtime의
    // ORTCHAR_T와 그대로 맞으므로 별도 변환이 필요 없다.
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

    input_name_ = session_->GetInputNameAllocated(0, allocator_).get();
    output_name_ = session_->GetOutputNameAllocated(0, allocator_).get();
}

const std::string& Detector::class_name(int class_id) const {
    static const std::string unknown = "unknown";
    const auto& names = coco_names();
    if (class_id < 0 || static_cast<size_t>(class_id) >= names.size()) return unknown;
    return names[class_id];
}

std::vector<Detection> Detector::detect(const Image& image, double* inference_ms) {
    LetterboxInfo info;
    std::vector<float> tensor = letterbox_to_tensor(image, options_.imgsz, info);

    const std::array<int64_t, 4> shape = {1, 3, options_.imgsz, options_.imgsz};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(memory, tensor.data(), tensor.size(),
                                                       shape.data(), shape.size());

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};

    const auto started = std::chrono::steady_clock::now();
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
    const auto elapsed = std::chrono::steady_clock::now() - started;
    if (inference_ms != nullptr) {
        *inference_ms = std::chrono::duration<double, std::milli>(elapsed).count();
    }

    // YOLOv8/YOLO11 출력: [1, 4 + num_classes, num_anchors]
    const float* data = outputs[0].GetTensorData<float>();
    const auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (out_shape.size() != 3) {
        throw std::runtime_error("예상과 다른 출력 형태입니다. YOLOv8/YOLO11 ONNX가 맞는지 확인하세요.");
    }
    const int64_t channels = out_shape[1];
    const int64_t anchors = out_shape[2];
    const int64_t num_classes = channels - 4;

    std::vector<Detection> candidates;
    for (int64_t a = 0; a < anchors; ++a) {
        int best_class = -1;
        float best_score = options_.conf_threshold;
        for (int64_t c = 0; c < num_classes; ++c) {
            const float score = data[(4 + c) * anchors + a];
            if (score > best_score) {
                best_score = score;
                best_class = static_cast<int>(c);
            }
        }
        if (best_class < 0) continue;

        // 모델은 letterbox 좌표계의 중심점과 크기를 낸다. 패딩을 빼고 배율로 나눠
        // 원본 이미지 좌표로 되돌린다.
        const float cx = data[0 * anchors + a];
        const float cy = data[1 * anchors + a];
        const float w = data[2 * anchors + a];
        const float h = data[3 * anchors + a];

        Detection det;
        det.x0 = (cx - w * 0.5f - info.pad_x) / info.scale;
        det.y0 = (cy - h * 0.5f - info.pad_y) / info.scale;
        det.x1 = (cx + w * 0.5f - info.pad_x) / info.scale;
        det.y1 = (cy + h * 0.5f - info.pad_y) / info.scale;
        det.score = best_score;
        det.class_id = best_class;
        candidates.push_back(det);
    }

    return non_max_suppression(std::move(candidates), options_.iou_threshold);
}

}  // namespace yolo
