cmake_minimum_required(VERSION 3.30)

project(Collective)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SOURCE_DIR "src")
set(INCLUDE_DIR "include")

set(SOURCES
  ${SOURCE_DIR}/main.cpp
  ${SOURCE_DIR}/embedding/EmbeddingLayer.cpp
  ${SOURCE_DIR}/embedding/PositionalEncoding.cpp
  ${SOURCE_DIR}/gpt/GPT.cpp
  ${SOURCE_DIR}/matrix/Matrix.cpp
  ${SOURCE_DIR}/tokenization/Tokenizer.cpp
  ${SOURCE_DIR}/transformer/TransformerBlock.cpp
)

set(HEADERS
  ${INCLUDE_DIR}/embedding/EmbeddingLayer.hpp
  ${INCLUDE_DIR}/embedding/PositionalEncoding.hpp
  ${INCLUDE_DIR}/gpt/GPT.hpp
  ${INCLUDE_DIR}/matrix/Matrix.hpp
  ${INCLUDE_DIR}/tokenization/Tokenizer.hpp
  ${INCLUDE_DIR}/transformer/TransformerBlock.hpp
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
