cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)
project(kepler_metal)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(${CMAKE_CURRENT_LIST_DIR}/metal-cpp)
add_subdirectory(src)
