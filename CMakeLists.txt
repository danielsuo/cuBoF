cmake_minimum_required(VERSION 3.1)

project(cuBoF CXX)

find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

if (APPLE)
  set(OpenCV_DIR "/usr/local/Cellar/opencv3/3.0.0/share/OpenCV")
endif()
find_package(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/lib)

set(VLROOT ${CMAKE_CURRENT_SOURCE_DIR}/lib/vlfeat)
include_directories(${VLROOT})
link_directories(${VLROOT}/bin/glnxa64)
MESSAGE(${VLROOT}/bin/glnxa64)

add_custom_target(
  VLLIB
  COMMAND make
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/vlfeat
)

# Add cuSIFT
add_subdirectory (lib/cuSIFT)

set(SOURCES
    main.cpp
    cuBoF.cpp
    utils.cpp
)

add_library (cuBoF ${SOURCES})
target_compile_options(cuBoF PRIVATE "-std=c++11")
target_link_libraries(cuBoF vl cuSIFT ${OpenCV_LIBS})
add_dependencies(cuBoF VLLIB)
target_include_directories (cuBoF PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})