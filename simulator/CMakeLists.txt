# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
cmake_minimum_required(VERSION 3.9)
SET(CMAKE_CXX_COMPILER "g++")
SET(CMAKE_CXX_STANDARD 17)

option(OPENCV "Enable OpenCV build" OFF)
option(GPU "Enable CUDA build" OFF)
option(BUILD_TESTS "Enable test target build" OFF)
option(CONFIDENTIAL_FEATURES "Build SAPEON internal features" OFF)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Manually add numpy include directory
set(NUMPY_INCLUDE_DIR /usr/lib/python3/dist-packages/numpy/core/include)
include_directories(${NUMPY_INCLUDE_DIR})

# Include directories for Python and pybind11
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIRS})

if(GPU)
    add_definitions(-DGPU)
    project(aix_simulator LANGUAGES CUDA CXX)
    find_package(CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
else()
    project(aix_simulator LANGUAGES CXX)
endif()

if(OPENCV)
    add_definitions(-DOPENCV)
    find_package(OpenCV)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

include_directories(
    include
    third_party/argparse/include
    third_party/libnpy/include
    third_party/expected/include
    third_party/stb/include
    third_party/gsl-lite/include
    third_party/nlohmann/include
)

file(GLOB SOURCES
    "src/backends/backend.cpp"
    "src/backends/backend_input_helper.cpp"
    "src/backends/cpu.cpp"
    "src/backends/delegate/*.cpp"
    "src/calibration/*.cpp"
    "src/core/*.cpp"
    "src/core/*.cc"
    "src/cpu/acts/*.cpp"
    "src/cpu/ops/*.cpp"
    "src/cpu/common/*.cpp"
    "src/network/*.cpp"
    "src/progress/*.cpp"
    "src/*.cpp"
    "src/x220/*.cpp"
    "src/x220/ops/*.cpp"
    "src/x330/*.cpp"
    "src/x330/ops/*.cpp"
    "src/x330/acts/*.cpp"
    "src/dump/*.cpp"
    "src/parser/*.cpp"
)

if(GPU)
    file(GLOB SOURCES
        ${SOURCES}
        "src/backends/cudnn.cpp"
        "src/cudnn/acts/*.cpp"
        "src/cudnn/ops/*.cpp"
        "src/cudnn/common/*.cpp"
    )
    file(GLOB SOURCES_CUDA
        "src/cudnn/acts/*.cu"
        "src/cudnn/ops/*.cu"
        "src/cudnn/common/*.cu"
    )
endif()

SET(BASE_LIBS
    protobuf
    pthread
    glog
    gflags
)
if(OPENCV)
    SET(BASE_LIBS
        ${BASE_LIBS}
        ${OpenCV_LIBS}
    )
endif()
if(GPU)
    SET(BASE_LIBS
        ${BASE_LIBS}
        cudnn
        cublas
        nsync_cpp
    )
endif()

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fopenmp -fsanitize=address -fno-omit-frame-pointer")
if(GPU)
    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr -include algorithm -fsanitize=address")
endif()

SET(TARGET_SHARED_LIB spsim)
SET(TARGET_BIN simulator)
SET(TARGET_CPU_TEST_BIN test_cpu)
SET(TARGET_CUDA_TEST_BIN test_gpu)

add_library(${TARGET_SHARED_LIB}
SHARED
    ${SOURCES}
    "src/simulator_api.cpp"
)

set_target_properties(${TARGET_SHARED_LIB}
PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(${TARGET_SHARED_LIB}
PRIVATE
    ${BASE_LIBS}
)

if(NOT BUILD_TESTS)
    if(GPU)
        cuda_add_executable(${TARGET_BIN}
            "src/simulator.cpp"
            ${SOURCES}
            ${SOURCES_CUDA}
        )
        set_property(TARGET ${TARGET_BIN} PROPERTY CUDA_STANDARD 14)
    else()
        add_executable(${TARGET_BIN}
            "src/simulator.cpp"
        )
    endif()

    target_link_libraries(${TARGET_BIN}
    PUBLIC
        ${TARGET_SHARED_LIB}
        glog
    )
else()
    enable_testing()
    find_package(GTest REQUIRED)
    include_directories(tests)
    set(TEST_SRC_DIR "tests")
    file(GLOB TEST_SOURCES
        "${TEST_SRC_DIR}/cpu/*.cpp"
        "${TEST_SRC_DIR}/cpu/tests/*.cpp"
        "${TEST_SRC_DIR}/arguments/*.cpp"
    )
    if(GPU)
        cuda_add_executable(${TARGET_CUDA_TEST_BIN}
            "${TEST_SRC_DIR}/test_main.cpp"
            "${TEST_SRC_DIR}/cudnn/test_cudnn_operation.cu"
            ${TEST_SOURCES}
            ${SOURCES}
            ${SOURCES_CUDA}
        )

        set_property(
        TARGET
            ${TARGET_CUDA_TEST_BIN} 
        PROPERTY
            CUDA_STANDARD 14
        )
        target_link_libraries(${TARGET_CUDA_TEST_BIN}
            gtest
            ${BASE_LIBS}
        )
    else()
        add_executable(${TARGET_CPU_TEST_BIN}
            "${TEST_SRC_DIR}/test_main.cpp"
            "${TEST_SRC_DIR}/test_simulator_api.cpp"
            # ${TEST_SOURCES}
        )
        target_link_libraries(${TARGET_CPU_TEST_BIN}
            gtest
            glog
            gflags
            pthread
            dl
            ${TARGET_SHARED_LIB}
        )
        target_compile_definitions(${TARGET_CPU_TEST_BIN}
        PRIVATE
            PROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        )
    endif()
endif()

include(python/pyspsim.cmake)
if(CONFIDENTIAL_FEATURES)
    include(confidential/confidential.cmake)
endif()
