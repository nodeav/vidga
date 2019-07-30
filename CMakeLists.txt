cmake_minimum_required(VERSION 3.12)
project(Vidga)

set(CMAKE_CXX_STANDARD 14)
set(PROJECT_NAME Vidga)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME}
        main.cpp
        interfaces/ga/ga.cpp
        interfaces/ga/ga.h
        interfaces/shape/shape.cpp
        interfaces/shape/shape.h
        interfaces/chromosome/chromosome.cpp
        interfaces/chromosome/chromosome.h
        classes/shapes/circle.cpp classes/shapes/circle.h
        )

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
