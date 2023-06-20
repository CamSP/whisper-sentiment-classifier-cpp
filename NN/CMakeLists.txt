cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(NNPredict)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package(Torch REQUIRED)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_executable(main src/main.cpp
                    include/network.h
                    )

target_include_directories(main PRIVATE include)
target_link_libraries(main ${TORCH_LIBRARIES}
                            ${MPI_CXX_LIBRARIES})

set_property(TARGET main PROPERTY CXX_STANDARD 17)

add_executable(process src/process.cpp
                    include/network.h
                    )

target_include_directories(process PRIVATE include)
target_link_libraries(process ${TORCH_LIBRARIES})

set_property(TARGET process PROPERTY CXX_STANDARD 17)
