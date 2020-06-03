cmake_minimum_required(VERSION 3.4 )
project(cifar)

### LibTorchPath ###
include("~/libtorch/share/cmake/Torch/TorchConfig.cmake")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

