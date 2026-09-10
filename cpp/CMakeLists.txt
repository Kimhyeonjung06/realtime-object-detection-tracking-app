cmake_minimum_required(VERSION 3.16)
project(yolo_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ONNX Runtime 배포본의 압축을 푼 경로를 지정한다.
#   cmake -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime-win-x64-1.29.0
set(ONNXRUNTIME_ROOT "" CACHE PATH "Root of an extracted ONNX Runtime release")
if(NOT ONNXRUNTIME_ROOT)
  message(FATAL_ERROR
    "ONNXRUNTIME_ROOT 가 필요합니다. https://github.com/microsoft/onnxruntime/releases 에서 "
    "플랫폼에 맞는 배포본을 받아 압축을 푼 뒤 그 경로를 지정하세요.")
endif()

find_library(ONNXRUNTIME_LIB
  NAMES onnxruntime
  PATHS "${ONNXRUNTIME_ROOT}/lib"
  REQUIRED
  NO_DEFAULT_PATH)

add_executable(yolo_infer
  src/main.cpp
  src/image_io.cpp
  src/yolo_detector.cpp)

target_include_directories(yolo_infer PRIVATE "${ONNXRUNTIME_ROOT}/include")
# 벤더링한 헤더는 SYSTEM으로 취급해 우리 코드의 경고만 남긴다.
target_include_directories(yolo_infer SYSTEM PRIVATE third_party)

target_link_libraries(yolo_infer PRIVATE "${ONNXRUNTIME_LIB}")

if(MSVC)
  target_compile_options(yolo_infer PRIVATE /W4 /utf-8)
else()
  target_compile_options(yolo_infer PRIVATE -Wall -Wextra)
  # MinGW 런타임(libstdc++, libgcc, libwinpthread)을 정적으로 넣는다.
  # 그러지 않으면 실행 파일이 툴체인 DLL을 찾지 못해 다른 PC에서 바로 죽는다.
  # -municode는 wmain 진입점을 쓰기 위해 필요하다.
  if(WIN32)
    target_link_options(yolo_infer PRIVATE -static -municode)
  endif()
endif()

# 공유 라이브러리를 실행 파일 옆에 놓아 바로 실행되게 한다.
if(WIN32)
  add_custom_command(TARGET yolo_infer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
      "$<TARGET_FILE_DIR:yolo_infer>")
endif()
