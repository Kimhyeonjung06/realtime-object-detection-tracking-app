// ONNX Runtime C++ API로 YOLO ONNX 모델을 실행하는 탐지기.
// 후처리(디코딩 + NMS)도 직접 구현한다. 파이썬 런타임 없이 모델 하나와 DLL만으로
// 도는 실행 파일을 만드는 것이 목적이다.
#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "image_io.hpp"

namespace yolo {

struct Detection {
    float x0, y0, x1, y1;  // 원본 이미지 좌표
    float score;
    int class_id;
};

struct Options {
    int imgsz = 640;
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
    int threads = 0;  // 0이면 ORT 기본값
};

class Detector {
public:
    Detector(const std::filesystem::path& model_path, const Options& options);

    // 한 장을 추론한다. inference_ms에는 전·후처리를 제외한 세션 실행 시간만 담긴다.
    std::vector<Detection> detect(const Image& image, double* inference_ms = nullptr);

    const std::string& class_name(int class_id) const;

private:
    Options options_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string input_name_;
    std::string output_name_;
};

}  // namespace yolo
