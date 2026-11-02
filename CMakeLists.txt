cmake_minimum_required(VERSION 3.21)
project(mathlib CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Single-config generators leave this empty, which also leaves `check`'s
# `-C $<CONFIG>` without a value.
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE
   AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# compile_commands.json for clangd, so no separate tool is needed.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(mathlib INTERFACE)
target_include_directories(mathlib INTERFACE ${PROJECT_SOURCE_DIR})
target_compile_features(mathlib INTERFACE cxx_std_23)

# Off by default when pulled in with add_subdirectory or FetchContent.
option(MATHLIB_BUILD_TESTS "Build the mathlib tests" ${PROJECT_IS_TOP_LEVEL})

if(MATHLIB_BUILD_TESTS)
  set(TEST_SOURCES
    test.cpp
  )

  enable_testing()
  set(CTEST_TEST_TIMEOUT 30)

  foreach(test_src IN LISTS TEST_SOURCES)
    # Prefixed because CTest reserves the bare target name `test`.
    get_filename_component(test_name "${test_src}" NAME_WE)
    set(test_name "mathlib_${test_name}")
    add_executable("${test_name}" "${test_src}")
    target_link_libraries("${test_name}" PRIVATE mathlib)

    if(MSVC)
      target_compile_options("${test_name}" PRIVATE /W4)
    else()
      target_compile_options("${test_name}" PRIVATE -Wall -Wextra)
    endif()

    add_test(NAME "${test_name}" COMMAND "${test_name}")
  endforeach()

  add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C $<CONFIG>
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  )
endif()
