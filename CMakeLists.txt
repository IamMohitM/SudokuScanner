cmake_minimum_required(VERSION 3.17)
project(SudokuSolver)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
add_executable(SudokuSolver main.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(SudokuSolver ${OpenCV_LIBS})