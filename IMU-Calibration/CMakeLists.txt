cmake_minimum_required(VERSION 3.10)
project(UKF_Orientation)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Eigen3 REQUIRED)

# Find matio library for reading MATLAB files
find_library(MATIO_LIBRARY NAMES matio)
find_path(MATIO_INCLUDE_DIR NAMES matio.h)

if(NOT MATIO_LIBRARY OR NOT MATIO_INCLUDE_DIR)
    message(WARNING "matio library not found. You may need to install it:")
    message("  Ubuntu/Debian: sudo apt-get install libmatio-dev")
    message("  macOS: brew install libmatio")
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${MATIO_INCLUDE_DIR}
)

# Source files
set(SOURCES
    src/ukf.cpp
    src/main.cpp
)

# Create executable
add_executable(ukf_orientation ${SOURCES})

# Link libraries
target_link_libraries(ukf_orientation
    ${MATIO_LIBRARY}
    Eigen3::Eigen
)

# Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(ukf_orientation PRIVATE -Wall -Wextra -O3)
endif()

# Installation
install(TARGETS ukf_orientation DESTINATION bin)

# Print configuration info
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Eigen3 version: ${EIGEN3_VERSION}")
message(STATUS "Matio library: ${MATIO_LIBRARY}")
message(STATUS "Matio include: ${MATIO_INCLUDE_DIR}")