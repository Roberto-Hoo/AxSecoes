cmake_minimum_required(VERSION 3.17)
project(AxSecoes)

set(CMAKE_CXX_STANDARD 14)

add_executable(AxSecoes main.cpp)
#Acesso ao GSL-GNU
target_link_libraries(AxSecoes -lgsl -lgslcblas -lm)
#Acesso ao OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp")
set(SOURCE_FILES main.cpp)
add_executable(openmp_test_clion ${SOURCE_FILES})