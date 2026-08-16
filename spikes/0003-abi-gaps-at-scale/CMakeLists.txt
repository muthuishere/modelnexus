# Spike 0003 — builds ONE throwaway executable against the same prebuilt
# llama.cpp the bridge links. Deliberately not wired into core/CMakeLists.txt:
# a spike is evidence, not a build target.
cmake_minimum_required(VERSION 3.18)
project(spike0003 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLB_HEADERS_DIR "" CACHE PATH "llama.cpp source root (headers)")
set(LLB_PREBUILT_DIR "" CACHE PATH "directory holding libllama / libllama-common")

add_executable(spike spike.cpp)

target_include_directories(spike PRIVATE
    "${LLB_HEADERS_DIR}/include"
    "${LLB_HEADERS_DIR}/ggml/include"
    "${LLB_HEADERS_DIR}/common"
    "${LLB_HEADERS_DIR}/vendor"
)

find_library(SPIKE_LIB_LLAMA  NAMES llama        PATHS "${LLB_PREBUILT_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(SPIKE_LIB_COMMON NAMES llama-common PATHS "${LLB_PREBUILT_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(SPIKE_LIB_GGML   NAMES ggml-base    PATHS "${LLB_PREBUILT_DIR}" NO_DEFAULT_PATH REQUIRED)
target_link_libraries(spike PRIVATE "${SPIKE_LIB_LLAMA}" "${SPIKE_LIB_COMMON}" "${SPIKE_LIB_GGML}")

if(APPLE)
    set_target_properties(spike PROPERTIES
        BUILD_RPATH "${LLB_PREBUILT_DIR}" MACOSX_RPATH TRUE)
elseif(UNIX)
    set_target_properties(spike PROPERTIES BUILD_RPATH "${LLB_PREBUILT_DIR}")
endif()
