cmake_minimum_required(VERSION 3.12)
project(Vidga)

set(CMAKE_CXX_STANDARD 14)
set(PROJECT_NAME Vidga)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Ofast -march=native -mtune=native")

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

include_directories(".")

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
        classes/renderers/opencvRenderer.h util.h
        classes/SerDes/SerDes.cpp
        classes/SerDes/SerDes.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} UTIL Threads::Threads)


#set(PROTO_TARGET_NAME prototype)
#include_directories(prototype)
#add_executable(${PROTO_TARGET_NAME} prototype/main.cpp)