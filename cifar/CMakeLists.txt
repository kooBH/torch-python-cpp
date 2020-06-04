cmake_minimum_required(VERSION 3.4 )
project(cifar)

### LibTorchPath ###
include("~/libtorch/share/cmake/Torch/TorchConfig.cmake")

### opencv
list(APPEND INCL
  #  ${CMAKE_CURRENT_SOURCE_DIR}/lib/opencv4
  /usr/local/include/opencv4
)

list(APPEND LINKLIBS 
  /usr/local/lib/libopencv_core.so.4.2.0
  /usr/local/lib/libopencv_highgui.so.4.2.0
  /usr/local/lib/libopencv_imgproc.so.4.2.0
  /usr/local/lib/libopencv_imgcodecs.so.4.2.0
  #  opencv_core
  # opencv_highgui
  # opencv_imgproc
  )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} main.cpp)
target_include_directories(${PROJECT_NAME}	PRIVATE	${INCL}  )
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${LINKLIBS} )
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)



