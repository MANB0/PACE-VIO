cmake_minimum_required(VERSION 3.16)
project(t2_isam2_backend LANGUAGES CXX)

find_package(GTSAM REQUIRED)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG QUIET)
set(T2_PYBIND11_SOURCE_DIR "" CACHE PATH
    "Optional pybind11 source tree used when a CMake package is unavailable")
if(NOT pybind11_FOUND)
  if(T2_PYBIND11_SOURCE_DIR AND EXISTS "${T2_PYBIND11_SOURCE_DIR}/CMakeLists.txt")
    add_subdirectory("${T2_PYBIND11_SOURCE_DIR}" pybind11 EXCLUDE_FROM_ALL)
  elseif(EXISTS "/home/admin1/gtsam/wrap/pybind11/CMakeLists.txt")
    # Backward-compatible local development fallback. Portable builds should
    # install pybind11 or pass T2_PYBIND11_SOURCE_DIR explicitly.
    add_subdirectory(/home/admin1/gtsam/wrap/pybind11 pybind11 EXCLUDE_FROM_ALL)
  else()
    message(FATAL_ERROR
      "pybind11 was not found. Install it in the active Python environment "
      "or pass -DT2_PYBIND11_SOURCE_DIR=/path/to/pybind11")
  endif()
endif()

add_executable(t2_isam2_runner t2_isam2_runner.cpp)
target_compile_features(t2_isam2_runner PRIVATE cxx_std_17)
target_link_libraries(t2_isam2_runner PRIVATE gtsam)

add_executable(test_t2_factors test_t2_factors.cpp)
target_compile_features(test_t2_factors PRIVATE cxx_std_17)
target_link_libraries(test_t2_factors PRIVATE gtsam)

pybind11_add_module(t2_isam2_backend MODULE t2_isam2_pybind.cpp)
target_compile_features(t2_isam2_backend PRIVATE cxx_std_17)
target_link_libraries(t2_isam2_backend PRIVATE gtsam)
set_target_properties(t2_isam2_backend PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)

enable_testing()
add_test(NAME t2_isam2_factors COMMAND test_t2_factors)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(t2_isam2_runner PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(test_t2_factors PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(t2_isam2_backend PRIVATE -Wall -Wextra -Wpedantic)
endif()
