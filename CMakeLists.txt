cmake_minimum_required(VERSION 3.30)

project(Collective)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SOURCE_DIR "src")
set(INCLUDE_DIR "include")

set(SOURCES
  ${SOURCE_DIR}/main.cpp
  ${SOURCE_DIR}/embedding/embedding-layer.cpp
  ${SOURCE_DIR}/tokenization/tokenizer.cpp
)

set(HEADERS
  ${INCLUDE_DIR}/embedding/embedding-layer.hpp
  ${INCLUDE_DIR}/tokenization/tokenizer.hpp
)

include_directories(${INCLUDE_DIR})

add_executable(${PROJECT_NAME}
  ${SOURCES}
  ${HEADERS}
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /WX)
endif()
