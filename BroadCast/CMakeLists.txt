cmake_minimum_required(VERSION 3.3)
project(mlpMpi C)
set(CMAKE_C_STANDARD 99)
find_package(MPI)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
add_executable(mlpMpi mlpMpi.c)
target_link_libraries(mlpMpi ${MPI_C_LIBRARIES} m)
