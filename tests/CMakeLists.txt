cmake_minimum_required(VERSION 3.18 FATAL_ERROR)



#mainline tests

ConfigureGPUErrExecutable(log_test "${CMAKE_CURRENT_SOURCE_DIR}/src/log_test.cu" "${GPU_ERR_TESTS_BINARY_DIR}")
