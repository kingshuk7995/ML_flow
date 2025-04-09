cmake_minimum_required(VERSION 3.14)
project(MLflow VERSION 0.1 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable OpenMP (optional)
find_package(OpenMP)

# === BLAS Support (commented out) ===
# Find system BLAS (e.g., OpenBLAS or system LAPACK)
# find_package(BLAS)

# === If using BLAS, enable Eigen to use BLAS-backed solvers ===
# if(BLAS_FOUND)
#     message(STATUS "BLAS library found: ${BLAS_LIBRARIES}")
#     target_compile_definitions(MLflow INTERFACE
#         EIGEN_USE_BLAS
#         EIGEN_USE_LAPACKE
#     )
#     target_link_libraries(MLflow INTERFACE ${BLAS_LIBRARIES})
# endif()

# Define MLflow header-only interface library
add_library(MLflow INTERFACE)

# Include Eigen and MLflow headers
target_include_directories(MLflow INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen3>
)

# Copy datasets to build directory
set(RESOURCE_DIRS datasets)
foreach(dir IN LISTS RESOURCE_DIRS)
    file(COPY ${CMAKE_SOURCE_DIR}/${dir} DESTINATION ${CMAKE_BINARY_DIR})
endforeach()

# Add test executables from tests/*.cpp
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE MLflow)

    # Link OpenMP to each test if available
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endforeach()
