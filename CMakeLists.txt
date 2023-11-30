cmake_minimum_required(VERSION 3.0.0)
project(CNN VERSION 0.1.0 LANGUAGES C)
set(CMAKE_C_STANDARD 99)

include(CTest)
enable_testing()

add_executable(CNN main.c)
target_link_libraries(CNN PRIVATE m)