#include "image_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;

namespace yolo {
namespace {

// stb는 좁은 문자열 경로만 받는다. 유니코드 경로를 살리기 위해 파일을 직접 열어
// FILE* 기반 API로 넘긴다.
std::FILE* open_file(const fs::path& path, bool write) {
#ifdef _WIN32
    return _wfopen(path.c_str(), write ? L"wb" : L"rb");
#else
    return std::fopen(path.c_str(), write ? "wb" : "rb");
#endif
}

void write_to_file(void* context, void* data, int size) {
    std::fwrite(data, 1, static_cast<size_t>(size), static_cast<std::FILE*>(context));
}

}  // namespace

std::string display(const fs::path& path) {
    try {
        return path.string();
    } catch (const std::exception&) {
        return "<path>";
    }
}

Image load_image(const fs::path& path) {
    std::FILE* file = open_file(path, /*write=*/false);
    if (file == nullptr) return {};

    Image image;
    int channels = 0;
    uint8_t* data = stbi_load_from_file(file, &image.width, &image.height, &channels, 3);
    std::fclose(file);
    if (data == nullptr) return {};

    image.rgb.assign(data, data + static_cast<size_t>(image.width) * image.height * 3);
    stbi_image_free(data);
    return image;
}

bool save_jpeg(const fs::path& path, const Image& image, int quality) {
    std::FILE* file = open_file(path, /*write=*/true);
    if (file == nullptr) return false;
    const int ok = stbi_write_jpg_to_func(write_to_file, file, image.width, image.height, 3,
                                          image.rgb.data(), quality);
    std::fclose(file);
    return ok != 0;
}

std::vector<float> letterbox_to_tensor(const Image& image, int size, LetterboxInfo& info) {
    info.scale = std::min(static_cast<float>(size) / image.width,
                          static_cast<float>(size) / image.height);
    const int new_w = static_cast<int>(std::round(image.width * info.scale));
    const int new_h = static_cast<int>(std::round(image.height * info.scale));
    info.pad_x = (size - new_w) / 2;
    info.pad_y = (size - new_h) / 2;

    // 파이썬 쪽 전처리와 맞추기 위해 빈 영역은 114로 채운다.
    const float gray = 114.0f / 255.0f;
    std::vector<float> tensor(static_cast<size_t>(3) * size * size, gray);

    const size_t plane = static_cast<size_t>(size) * size;
    for (int y = 0; y < new_h; ++y) {
        // 최근접 이웃 샘플링. 전처리 비용을 낮게 유지하는 것이 목적이라 보간은 쓰지 않는다.
        const int src_y = std::min(image.height - 1, static_cast<int>(y / info.scale));
        for (int x = 0; x < new_w; ++x) {
            const int src_x = std::min(image.width - 1, static_cast<int>(x / info.scale));
            const uint8_t* px = &image.rgb[(static_cast<size_t>(src_y) * image.width + src_x) * 3];
            const size_t dst = static_cast<size_t>(y + info.pad_y) * size + (x + info.pad_x);
            tensor[dst] = px[0] / 255.0f;
            tensor[plane + dst] = px[1] / 255.0f;
            tensor[2 * plane + dst] = px[2] / 255.0f;
        }
    }
    return tensor;
}

void draw_rect(Image& image, int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b,
               int thickness) {
    x0 = std::clamp(x0, 0, image.width - 1);
    x1 = std::clamp(x1, 0, image.width - 1);
    y0 = std::clamp(y0, 0, image.height - 1);
    y1 = std::clamp(y1, 0, image.height - 1);

    auto put = [&](int x, int y) {
        if (x < 0 || y < 0 || x >= image.width || y >= image.height) return;
        uint8_t* px = &image.rgb[(static_cast<size_t>(y) * image.width + x) * 3];
        px[0] = r;
        px[1] = g;
        px[2] = b;
    };

    for (int t = 0; t < thickness; ++t) {
        for (int x = x0; x <= x1; ++x) {
            put(x, y0 + t);
            put(x, y1 - t);
        }
        for (int y = y0; y <= y1; ++y) {
            put(x0 + t, y);
            put(x1 - t, y);
        }
    }
}

std::vector<fs::path> collect_images(const fs::path& path) {
    static const std::vector<std::string> kExtensions = {".jpg", ".jpeg", ".png", ".bmp"};
    std::vector<fs::path> files;

    if (fs::is_directory(path)) {
        for (const auto& entry : fs::directory_iterator(path)) {
            if (!entry.is_regular_file()) continue;
            // 확장자는 ASCII이므로 좁은 문자열로 비교해도 안전하다.
            std::string ext = entry.path().extension().generic_string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (std::find(kExtensions.begin(), kExtensions.end(), ext) != kExtensions.end()) {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());
    } else if (fs::is_regular_file(path)) {
        files.push_back(path);
    }
    return files;
}

}  // namespace yolo
