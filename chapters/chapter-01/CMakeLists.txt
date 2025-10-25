
# Chapter 01: Vector Add (CPU & CUDA)

add_executable(vecadd_cpu vecadd_cpu.c)
target_link_libraries(vecadd_cpu PRIVATE pmpp_common)

add_executable(vecadd_cuda vecadd_cuda.cu)
target_link_libraries(vecadd_cuda PRIVATE pmpp_common CUDA::cudart)
set_target_properties(vecadd_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)