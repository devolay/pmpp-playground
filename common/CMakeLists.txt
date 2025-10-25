add_library(pmpp_common STATIC
        src/utils.c
        src/timer.c
)

target_include_directories(pmpp_common
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (NOT MSVC)
    target_compile_options(pmpp_common PRIVATE -O3)
endif()
