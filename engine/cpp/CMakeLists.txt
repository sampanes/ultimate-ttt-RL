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

# Native PUCT selection (#45a). A SECOND module rather than more surface on
# uttt_engine: this one mirrors search state, the other one is the rules, and
# a rules bug and a search bug should not share a build artifact.
pybind11_add_module(uttt_select uttt_select.cpp)

# Same optimization level, but float semantics are pinned. The output of this
# module is a discrete index chosen by comparing doubles, so a reassociation or
# an FMA contraction that changes a score's last bit can change WHICH CHILD IS
# SEARCHED. /fp:precise (and -ffp-contract=off elsewhere) is not caution, it is
# the parity requirement expressed to the compiler.
target_compile_options(uttt_select PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /W3 /fp:precise>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -Wall -ffp-contract=off>
)
