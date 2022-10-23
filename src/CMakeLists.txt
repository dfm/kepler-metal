add_custom_target(
    kernel_air
    COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/kernel.metal -o ${CMAKE_CURRENT_BINARY_DIR}/kernel.air
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernel.metal
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/kernel.air)
add_custom_target(
    kernel_metallib
    COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/kernel.air -o ${CMAKE_CURRENT_BINARY_DIR}/default.metallib
    DEPENDS kernel_air
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)

add_executable(demo main.cc)
add_dependencies(demo kernel_metallib)
target_link_libraries(demo
    "-framework Metal"
    "-framework MetalKit"
    "-framework AppKit"
    "-framework Foundation"
    "-framework QuartzCore"
)
