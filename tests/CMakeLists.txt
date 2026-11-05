# Test CMakeLists
add_executable(run_tests
    test_tensor.cpp
)

target_link_libraries(run_tests PRIVATE
    inference_engine
    gtest_main
)

add_test(NAME inference_tests COMMAND run_tests)