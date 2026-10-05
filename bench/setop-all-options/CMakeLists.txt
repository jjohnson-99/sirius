# Standalone libcudf harness for the EXCEPT ALL / INTERSECT ALL lowering choice.
# Not part of the Sirius build; configure it inside a Sirius pixi environment so
# it links the same libcudf.
cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

project(setop_all_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(cudf REQUIRED CONFIG)

add_executable(setop_all_bench setop_all_bench.cpp)
target_link_libraries(setop_all_bench PRIVATE cudf::cudf)
