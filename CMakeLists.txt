cmake_minimum_required(VERSION 3.20)
project(InferenceEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)
find_package(spdlog CONFIG REQUIRED)

option(WITH_AVX2 "Enable AVX2 optimizations" ON)

add_library(inference_engine STATIC
    src/core/tensor.cpp
    src/core/device.cpp
    src/gguf/gguf_parser.cpp
    src/gguf/quantization.cpp
    src/ops/ops.cpp
    src/ops/attention.cpp
    src/ops/normalization.cpp
    src/models/transformer.cpp
    src/models/model.cpp
    src/models/kv_cache.cpp
)

target_include_directories(inference_engine PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(inference_engine PUBLIC
    Threads::Threads
    spdlog::spdlog
)

if(WITH_AVX2)
    target_compile_definitions(inference_engine PUBLIC WITH_AVX2)
    target_compile_options(inference_engine PUBLIC $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma>)
endif()

install(TARGETS inference_engine
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
