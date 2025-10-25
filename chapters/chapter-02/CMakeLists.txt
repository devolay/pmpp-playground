# Chapter 02: Multidimensional Kernels

add_executable(rgb_to_gray rgb_to_gray.cu)
target_link_libraries(rgb_to_gray PRIVATE pmpp_common CUDA::cudart)
set_target_properties(rgb_to_gray PROPERTIES CUDA_SEPARABLE_COMPILATION ON)