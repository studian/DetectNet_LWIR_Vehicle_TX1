
file(GLOB detectnetConsoleSources *.cpp)
file(GLOB detectnetConsoleIncludes *.h )

cuda_add_executable(detectnet-lwir ${detectnetConsoleSources})
target_link_libraries(detectnet-lwir nvcaffe_parser nvinfer DetectNet_LWIR_Vehicle_TX1)
