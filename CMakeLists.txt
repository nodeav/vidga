cmake_minimum_required(VERSION 3.12)
project(Vidga)

set(CMAKE_CXX_STANDARD 14)

add_executable(Vidga
        main.cpp
        interfaces/ga/ga.cpp
        interfaces/ga/ga.h
        interfaces/shape/shape.cpp
        interfaces/shape/shape.h
        interfaces/chromosome/chromosome.cpp
        interfaces/chromosome/chromosome.h
        classes/shapes/circle.cpp classes/shapes/circle.h)