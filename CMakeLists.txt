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
        interfaces/individual/individual.cpp
        interfaces/individual/individual.h
        classes/shapes/circle.cpp classes/shapes/circle.h
        classes/individual/simpleIndividual.cpp classes/individual/simpleIndividual.h classes/population/simplePopulation.cpp classes/population/simplePopulation.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
