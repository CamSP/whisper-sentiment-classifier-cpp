cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(NNet)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package(Torch REQUIRED)

add_executable(main src/main.cpp
                    include/network.h
                    )

target_include_directories(main PRIVATE include)
target_link_libraries(main ${TORCH_LIBRARIES})

set_property(TARGET main PROPERTY CXX_STANDARD 17)
