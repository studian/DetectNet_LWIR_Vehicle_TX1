
file(GLOB v4l2ConsoleSources *.cpp)
file(GLOB v4l2ConsoleIncludes *.h )

add_executable(v4l2-console ${v4l2ConsoleSources})
target_link_libraries(v4l2-console DetectNet_LWIR_Vehicle_TX1)
