cmake_minimum_required(VERSION 3.15)
project(uttt_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(uttt_engine uttt_engine.cpp)

# Release optimizations
target_compile_options(uttt_engine PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /W3>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -Wall>
)
