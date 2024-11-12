find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
execute_process(COMMAND bash -c "pybind11-config --cmakedir | tr -d '\n'" OUTPUT_VARIABLE pybind11_DIR)
find_package(pybind11 REQUIRED)
find_package(Python REQUIRED COMPONENTS NumPy)
message("NumPy include path: ${Python_NumPy_INCLUDE_DIRS}")

set(TARGET_PYLIB pysimulator)
set(PYLIB_SRC_DIR ${CMAKE_SOURCE_DIR}/python)

pybind11_add_module(${TARGET_PYLIB}
  ${PYLIB_SRC_DIR}/spsim.cpp
  ${PYLIB_SRC_DIR}/calibrator.cpp
  ${PYLIB_SRC_DIR}/numpy.cpp)

target_link_libraries(${TARGET_PYLIB} PUBLIC ${TARGET_SHARED_LIB})
target_include_directories(${TARGET_PYLIB}
PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/python
  ${Python_NumPy_INCLUDE_DIRS})

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/sapeon)
set_target_properties(${TARGET_PYLIB}
  PROPERTIES
  SUFFIX ".so"
  OUTPUT_NAME "simulator"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/sapeon"
)
