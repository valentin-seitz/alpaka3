#
# Copyright 2026 Valentin Seitz
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(smallKernels LANGUAGES CXX)

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

# the benchmark requires OpenMP because it uses the executor cpuOmpBlockTasks and the queue kind ompFreeAgent
if(TARGET OpenMP::OpenMP_CXX)
    add_executable(smallKernels "src/smallKernels.cpp")
    target_link_libraries(smallKernels PUBLIC alpaka::alpaka)
    alpaka_finalize(smallKernels)
    add_test(NAME smallKernels COMMAND "smallKernels" 2 20 4096 4)
else()
    message(STATUS "smallKernels is not built because OpenMP is not available.")
endif()
