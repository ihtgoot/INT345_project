cmake_minimum_required(VERSION 3.10)

project(rpca_accelerated)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED PATHS ${pybind11_DIR} NO_DEFAULT_PATH)

add_library(rpca_module MODULE
    rpca_module.cpp
    rpca_solver.cpp
)

target_include_directories(rpca_module PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(rpca_module PRIVATE
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
    pybind11::module
)

set_target_properties(rpca_module PROPERTIES
    CXX_STANDARD 17
    PREFIX ""
    SUFFIX ".so"
)
