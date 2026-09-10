// 이미지 입출력과 전처리. OpenCV 없이 stb 단일 헤더 라이브러리만 사용한다.
// 임베디드 타깃에서는 무거운 영상 라이브러리를 통째로 올리기 어려운 경우가 많아,
// 의존성을 헤더 두 개로 제한했다.
//
// 경로는 std::string이 아니라 std::filesystem::path로 다룬다. Windows에서 좁은 문자열
// 경로는 시스템 코드페이지로 해석되어, ASCII가 아닌 경로가 섞이면 그대로 깨진다.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace yolo {

struct Image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;  // width * height * 3

    bool valid() const { return width > 0 && height > 0; }
};

// letterbox 결과를 원본 좌표로 되돌리는 데 필요한 정보
struct LetterboxInfo {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

Image load_image(const std::filesystem::path& path);
bool save_jpeg(const std::filesystem::path& path, const Image& image, int quality = 90);

// 비율을 유지한 채 정사각형 캔버스에 넣고 NCHW float[0,1] 텐서로 변환한다.
std::vector<float> letterbox_to_tensor(const Image& image, int size, LetterboxInfo& info);

void draw_rect(Image& image, int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b,
               int thickness = 2);

// 디렉터리면 내부 이미지 파일 목록을, 파일이면 그 파일 하나를 돌려준다.
std::vector<std::filesystem::path> collect_images(const std::filesystem::path& path);

// 콘솔 출력용. 변환할 수 없는 경로는 예외 대신 대체 문자열을 돌려준다.
std::string display(const std::filesystem::path& path);

}  // namespace yolo
