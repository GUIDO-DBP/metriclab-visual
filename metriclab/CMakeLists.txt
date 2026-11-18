cmake_minimum_required(VERSION 3.16)
project(metriclab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(metriclab_core
  src/vp_tree.cpp
  src/laesa.cpp
  src/trace.cpp
)
target_include_directories(metriclab_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(metriclab_core PRIVATE -Wall -Wextra -O2)

add_executable(oracle_test tests/oracle_test.cpp)
target_link_libraries(oracle_test PRIVATE metriclab_core)
target_compile_options(oracle_test PRIVATE -Wall -Wextra -O2)

add_executable(bench bench/bench_main.cpp)
target_link_libraries(bench PRIVATE metriclab_core)
target_compile_options(bench PRIVATE -Wall -Wextra -O2)

add_executable(search_cli apps/search_cli.cpp)
target_link_libraries(search_cli PRIVATE metriclab_core)
target_compile_options(search_cli PRIVATE -Wall -Wextra -O2)

set_target_properties(metriclab_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(metriclab_bridge SHARED apps/web_bridge.cpp)
target_link_libraries(metriclab_bridge PRIVATE metriclab_core)
set_target_properties(metriclab_bridge PROPERTIES OUTPUT_NAME metriclab)

enable_testing()
add_test(NAME oraculo COMMAND oracle_test)
