cmake_minimum_required(VERSION 3.12)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CXX_COMPILER "/usr/local/cuda/bin/nvcc")
project(Vidga CXX CUDA)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

include_directories(
        .
        cuda
        ${CUDA_INCLUDE_DIRS}
        ${OPENCV_INCLUDE_DIRS}
)

add_library(UTIL STATIC util.cpp util.h)

add_executable(${PROJECT_NAME}
        main.cpp
        interfaces/ga/ga.cpp
        interfaces/ga/ga.h
        interfaces/shape/shape.cpp
        interfaces/shape/shape.h
        interfaces/individual/individual.cpp
        interfaces/individual/individual.h
        classes/shapes/circle.cpp classes/shapes/circle.h
        classes/individual/simpleIndividual.cpp
        classes/individual/simpleIndividual.h
        classes/population/simplePopulation.cpp
        classes/population/simplePopulation.h
        classes/renderers/opencvRenderer.cpp
        classes/renderers/opencvRenderer.h
        util.h)

set(CUDA_LIB_NAME "cudacircles")
add_library(${CUDA_LIB_NAME} SHARED cuda/cudaCircles.cu cuda/cudaCircles.cu cuda/cudaCircles.cuh)
target_link_libraries(${CUDA_LIB_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} UTIL Threads::Threads ${CUDA_LIBRARIES} ${CUDA_LIB_NAME})

    #set(PROTO_TARGET_NAME prototype)
#include_directories(prototype)
#add_executable(${PROTO_TARGET_NAME} prototype/main.cpp)