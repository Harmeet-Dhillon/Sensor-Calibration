cmake_minimum_required(VERSION 3.16)
project(AutoCalib VERSION 1.0.0 LANGUAGES CXX)

# ── C++17 required for std::filesystem ──────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── OpenCV ───────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Install OpenCV 4.x and re-run cmake.")
endif()
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")

# ── Build type default ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── Compiler warnings ─────────────────────────────────────────────────────────
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(AutoCalib Wrapper.cpp)

target_include_directories(AutoCalib PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(AutoCalib PRIVATE ${OpenCV_LIBS})

# Link std::filesystem on older GCC
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(AutoCalib PRIVATE stdc++fs)
endif()

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS AutoCalib DESTINATION bin)

# ── Print summary ─────────────────────────────────────────────────────────────
message(STATUS "--------------------------------------")
message(STATUS "  Project  : ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "  Build    : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  OpenCV   : ${OpenCV_VERSION}")
message(STATUS "--------------------------------------")
