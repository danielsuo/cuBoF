cmake_minimum_required(VERSION 2.6)

find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

if (APPLE)
  set(OpenCV_DIR "/usr/local/Cellar/opencv3/3.0.0/share/OpenCV")
endif()
find_package(OpenCV REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/cuSIFT)

set(VLROOT ${CMAKE_CURRENT_SOURCE_DIR}/lib/vlfeat)
include_directories(${VLROOT})
link_directories(${VLROOT}/bin/glnxa64)

add_custom_target(
  VLLIB
  COMMAND make
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/vlfeat
)

set(SOURCES
    main.cpp
    cuBoF.cu
    utils.cpp
    lib/cuSIFT/cudaImage.cu
    lib/cuSIFT/cudaSiftH.cu
    lib/cuSIFT/extras/debug.h
)

cuda_add_executable(cuBoF ${SOURCES})
target_link_libraries(cuBoF vl ${OpenCV_LIBS})
add_dependencies(cuBoF VLLIB)