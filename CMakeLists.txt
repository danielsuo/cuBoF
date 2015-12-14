cmake_minimum_required(VERSION 2.6)

find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

if (APPLE)
  set(OpenCV_DIR "/usr/local/Cellar/opencv3/3.0.0/share/OpenCV")
endif()
find_package(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/lib/vlfeat)

ADD_CUSTOM_TARGET(
  vlfeat
  COMMAND make
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/vlfeat
)

set(SOURCES
    main.cpp
)

cuda_add_executable(cuBoF ${SOURCES})
target_link_libraries(cuBoF ${CMAKE_CURRENT_SOURCE_DIR}/lib/vlfeat/bin/glnxa64/libvl.so)
add_dependencies(cuBoF vlfeat)