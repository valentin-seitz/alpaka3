#
# Copyright 2025 René Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(ompOverhead LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
    option(alpaka_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)
    if(alpaka_USE_SOURCE_TREE)
        # Don't build the benchmarks recursively
        set(alpaka_BUILD_BENCHMARKS OFF)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/alpaka")
    else()
        find_package(alpaka REQUIRED)
    endif()
endif()

add_executable(ompOverhead src/ompOverhead.cpp)
target_link_libraries(ompOverhead PRIVATE alpaka::alpaka)
alpaka_finalize(ompOverhead)
set_target_properties(ompOverhead PROPERTIES FOLDER benchmarks/ompOverhead)

# short smoke run only, use script/run.sh for real measurements
add_test(NAME ompOverhead COMMAND ompOverhead -r 2 -m 1000)
