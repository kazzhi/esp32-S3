# The following lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
set(PARTITION_TABLE_CSV ${CMAKE_SOURCE_DIR}/partitions.csv)
#set(EXTRA_COMPONENT_DIRS ${CMAKE_SOURCE_DIR}/components/tensorflow_lite)
set(EXTRA_COMPONENT_DIRS
    ${CMAKE_SOURCE_DIR}/components/tensorflow_lite
    ${CMAKE_SOURCE_DIR}/components/esp-nn

)
include_directories(${CMAKE_SOURCE_DIR}/components/flatbuffers/include)

project(helloworld)

