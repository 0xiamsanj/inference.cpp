# Embed a PTX file into a generated C++ header as a string literal.
# Usage: cmake -DIN=file.ptx -DOUT=header.h -P EmbedPtx.cmake

if(NOT DEFINED IN OR NOT DEFINED OUT)
    message(FATAL_ERROR "EmbedPtx.cmake requires -DIN=<ptx> -DOUT=<header>")
endif()

file(READ "${IN}" PTX)

# Escape for a C string literal. Backslashes first, then quotes, then newlines.
string(REPLACE "\\" "\\\\" PTX "${PTX}")
string(REPLACE "\"" "\\\"" PTX "${PTX}")
string(REPLACE "\r" "" PTX "${PTX}")
string(REPLACE "\n" "\\n" PTX "${PTX}")

file(WRITE "${OUT}"
"// Auto-generated from cuda_kernels.cu -- do not edit.\n"
"#pragma once\n"
"static const char kCudaPtx[] =\n\"${PTX}\";\n")
