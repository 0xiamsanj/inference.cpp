cmake_minimum_required(VERSION 3.20)
project(InferenceEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")
endif()

# Build types
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find packages
find_package(Threads REQUIRED)

# Optional: CUDA backend.
# We do not require the CUDA toolkit: kernels are compiled to PTX with clang,
# then JIT-compiled by the installed NVIDIA driver at runtime (libcuda.so.1).
option(WITH_CUDA "Enable CUDA backend (clang PTX + driver JIT)" ON)

# Optional: SIMD
option(WITH_AVX2 "Enable AVX2 optimizations" ON)

# Dependency: spdlog (header-only logging)
find_package(spdlog CONFIG REQUIRED)

# Dependency: googletest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Main library
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
    src/models/model_cuda.cpp
    src/models/kv_cache.cpp
    src/sampling/sampler.cpp
    src/tokenizer/tokenizer.cpp
    src/ops/cuda_backend.cpp
)

target_include_directories(inference_engine PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(inference_engine PUBLIC
    Threads::Threads
    spdlog::spdlog
)

if(WITH_CUDA)
    find_program(CLANGXX NAMES clang++ clang)
    if(CLANGXX)
        set(CUDA_GPU_ARCH "sm_75" CACHE STRING "PTX target (GTX 1650 = sm_75)")

        set(CUDA_PTX_DIR "${CMAKE_BINARY_DIR}/generated")
        set(CUDA_PTX "${CUDA_PTX_DIR}/cuda_kernels.ptx")
        set(CUDA_PTX_HDR "${CUDA_PTX_DIR}/cuda_ptx_data.h")
        file(MAKE_DIRECTORY "${CUDA_PTX_DIR}")

        add_custom_command(
            OUTPUT "${CUDA_PTX}"
            COMMAND "${CLANGXX}" -x cuda --cuda-device-only -nocudainc -nocudalib
                    --cuda-gpu-arch=${CUDA_GPU_ARCH} -O3 -std=c++17 -S
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/cuda_kernels.cu"
                    -o "${CUDA_PTX}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/cuda_kernels.cu"
            COMMENT "Compiling CUDA kernels to PTX (${CUDA_GPU_ARCH})"
            VERBATIM
        )

        add_custom_command(
            OUTPUT "${CUDA_PTX_HDR}"
            COMMAND "${CMAKE_COMMAND}" "-DIN=${CUDA_PTX}" "-DOUT=${CUDA_PTX_HDR}"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedPtx.cmake"
            DEPENDS "${CUDA_PTX}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedPtx.cmake"
            COMMENT "Embedding PTX into the binary"
            VERBATIM
        )

        add_custom_target(cuda_ptx DEPENDS "${CUDA_PTX_HDR}")
        add_dependencies(inference_engine cuda_ptx)

        target_include_directories(inference_engine PRIVATE "${CUDA_PTX_DIR}")
        target_compile_definitions(inference_engine PUBLIC WITH_CUDA)
        target_link_libraries(inference_engine PUBLIC ${CMAKE_DL_LIBS})
        message(STATUS "CUDA backend: enabled (clang PTX, arch ${CUDA_GPU_ARCH})")
    else()
        message(WARNING "clang++ not found; CUDA backend disabled (CPU only).")
    endif()
endif()

if(WITH_AVX2)
    target_compile_definitions(inference_engine PUBLIC WITH_AVX2)
    target_compile_options(inference_engine PUBLIC $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma>)
endif()

# Executable
add_executable(inference_cli
    src/main.cpp
)
target_link_libraries(inference_cli PRIVATE inference_engine)

# Tests
enable_testing()
add_subdirectory(tests)

# Install
install(TARGETS inference_engine
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(TARGETS inference_cli RUNTIME DESTINATION bin)
# Diagnostic tool
add_executable(debug_logits tools/debug_logits.cpp)
target_link_libraries(debug_logits PRIVATE inference_engine)
