cmake_minimum_required(VERSION 3.13)
project(OpenCV)

set(CMAKE_CXX_STANDARD 17)

add_executable(OpenCV main.cpp )

cmake_minimum_required(VERSION 3.7)
find_package(OpenCV)
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( OpenCV ${OpenCV_LIBS} )



