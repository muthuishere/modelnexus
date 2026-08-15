# CMakeLists.txt for the modelnexus native bridge.
#
# Reused verbatim from mochallama, where it has shipped to Maven Central and npm.
# Orchestrated by ./build.sh here instead of Gradle; the two modes are unchanged.
#
# Builds ONE shared library — libllamabridge — our thin C wrapper that exposes
# a stable C ABI to Java (Panama FFM) over llama.cpp's C++ common_chat /
# common_sampler APIs.
#
# TWO modes (selected by ./build.sh):
#
#  1. PREBUILT (default) — `-DLLB_PREBUILT_DIR=<dir>` points at extracted
#     llama.cpp official release libs (libllama / libggml* / libllama-common).
#     We link the bridge against those by path and DO NOT compile llama.cpp.
#     This is the fast, no-OOM path. Headers come from `-DLLB_HEADERS_DIR`
#     (a shallow llama.cpp source checkout — headers only, never compiled).
#
#  2. SOURCE (fallback, `-Pnative=source`) — no LLB_PREBUILT_DIR; we
#     add_subdirectory() the vendored llama.cpp and build it from source
#     (the old behaviour). Kept so a bad upstream asset can't brick the build.
#     Use this when you want to rebuild everything from source yourself.
#
# Output: libllamabridge.<dylib|so|dll> in the CMake build dir; build.sh stages
# it next to the (prebuilt or freshly built) llama/ggml libs under
# core/dist/<platform-key>/.

cmake_minimum_required(VERSION 3.18)
project(llamabridge CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Headers root (llama.cpp source tree — used in BOTH modes for include paths).
set(LLB_HEADERS_DIR "${CMAKE_SOURCE_DIR}/third_party/llama.cpp"
    CACHE PATH "llama.cpp source root (headers only in prebuilt mode)")

add_library(llamabridge SHARED src/llamabridge.cpp)

target_include_directories(llamabridge
    PUBLIC  include
    PRIVATE "${LLB_HEADERS_DIR}/include"
            "${LLB_HEADERS_DIR}/ggml/include"
            "${LLB_HEADERS_DIR}/common"
            "${LLB_HEADERS_DIR}/vendor"
)

target_compile_definitions(llamabridge PRIVATE
    LLB_BRIDGE_VERSION="0.1.0"
    LLB_LLAMA_TAG="b9371"
)

# MSVC does not export DLL symbols by default (Unix exports everything). Without
# this, the bridge DLL loads but callers can't find llb_chat_create / llb_version
# etc. ("Symbol not found in libllamabridge"). Auto-export the whole extern-"C"
# surface so the Windows DLL matches the macOS/Linux default visibility.
set_target_properties(llamabridge PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(DEFINED LLB_PREBUILT_DIR AND NOT LLB_PREBUILT_DIR STREQUAL "")
    # -------------------- PREBUILT MODE --------------------
    message(STATUS "llamabridge: linking against PREBUILT llama.cpp in ${LLB_PREBUILT_DIR}")
    if(NOT EXISTS "${LLB_HEADERS_DIR}/include/llama.h")
        message(FATAL_ERROR "Headers not found at ${LLB_HEADERS_DIR} (need a shallow llama.cpp checkout for headers).")
    endif()
    find_library(LLB_LIB_LLAMA  NAMES llama        PATHS "${LLB_PREBUILT_DIR}" NO_DEFAULT_PATH REQUIRED)
    find_library(LLB_LIB_COMMON NAMES llama-common PATHS "${LLB_PREBUILT_DIR}" NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(llamabridge PRIVATE "${LLB_LIB_LLAMA}" "${LLB_LIB_COMMON}")
else()
    # -------------------- SOURCE MODE (fallback) --------------------
    message(STATUS "llamabridge: building llama.cpp FROM SOURCE (fallback mode)")
    set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS     OFF CACHE BOOL "" FORCE)
    set(LLAMA_CURL            OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS     ON  CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_COMMON    ON  CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_UI        OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_HTML      OFF CACHE BOOL "" FORCE)
    set(LLAMA_OPENSSL         OFF CACHE BOOL "" FORCE)
    # Single ggml-cpu for the host arch (NOT the split runtime-dispatch variants
    # that the prebuilt release packages ship) — keeps the runtime closure simple
    # and matches NativeLoader's load order. No OpenMP so Windows doesn't pull a
    # vcomp/libomp runtime dependency into the staged set.
    set(GGML_OPENMP          OFF CACHE BOOL "" FORCE)
    set(GGML_NATIVE          OFF CACHE BOOL "" FORCE)
    set(GGML_METAL            OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA             OFF CACHE BOOL "" FORCE)
    set(GGML_VULKAN           OFF CACHE BOOL "" FORCE)
    set(GGML_HIP              OFF CACHE BOOL "" FORCE)
    if(APPLE)
        set(GGML_ACCELERATE   ON  CACHE BOOL "" FORCE)
        set(GGML_BLAS         ON  CACHE BOOL "" FORCE)
        set(GGML_BLAS_VENDOR  "Apple" CACHE STRING "" FORCE)
    else()
        set(GGML_ACCELERATE   OFF CACHE BOOL "" FORCE)
        set(GGML_BLAS         OFF CACHE BOOL "" FORCE)
    endif()
    if(NOT EXISTS "${LLB_HEADERS_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "Vendored llama.cpp not found at ${LLB_HEADERS_DIR}. Clone it before building.")
    endif()
    add_subdirectory("${LLB_HEADERS_DIR}" llama-build EXCLUDE_FROM_ALL)
    target_link_libraries(llamabridge PRIVATE llama llama-common)
endif()

# Resolve sibling shared libs (llama, ggml*) at runtime from the bridge's own
# directory — that's where build.sh/CI stage them. Per-OS rpath syntax.
if(APPLE)
    set_target_properties(llamabridge PROPERTIES
        INSTALL_RPATH            "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
        MACOSX_RPATH             TRUE
    )
elseif(UNIX)
    set_target_properties(llamabridge PROPERTIES
        INSTALL_RPATH            "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()
