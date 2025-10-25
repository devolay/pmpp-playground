cmake_minimum_required(VERSION 3.22)
project(pmpp-playground LANGUAGES C CXX CUDA)

# Options
option(BUILD_TESTS "Build tests" OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# CUDA settings
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Nice warnings
if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wshadow -Wpedantic)
endif()

# Place binaries into bin/ for convenience
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(common)
add_subdirectory(chapters/chapter-01)

# Print summary
message(STATUS "Config Summary:")
message(STATUS "  C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "  CXX compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  CUDA compiler: ${CMAKE_CUDA_COMPILER}")