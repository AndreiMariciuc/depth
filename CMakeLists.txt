cmake_minimum_required(VERSION 3.23)
project(depthEstimation)

set(CMAKE_CXX_STANDARD 11)

#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--stack,10000000")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--heap,10000000")

find_package(OpenCV REQUIRED)

add_executable(depthEstimation main.cpp utils.h censusTransformation.cpp censusTransformation.h disparityCalculator.cpp disparityCalculator.h utils.cpp dp2DMultiBlocksDisparityCalculator.cpp dp2DMultiBlocksDisparityCalculator.h)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(depthEstimation ${OpenCV_LIBS})
