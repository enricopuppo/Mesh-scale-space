cmake_minimum_required(VERSION 3.2)
project(Mesh_scale_space)

add_executable(${PROJECT_NAME} main.cpp diff_geo.h diff_geo.cpp utilities.h utilities.cpp drawing_circle.h drawing_circle.cpp)
set_target_properties(${PROJECT_NAME}     PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
include_directories(${CMAKE_SOURCE_DIR}/libs/Eigen)
include_directories(${CMAKE_SOURCE_DIR}/libs/ann_1.1.2/include)
set(cinolib_DIR /Users/enricopuppo/Documents/GitHub/lib/cinolib)
set(CINOLIB_USES_OPENGL_GLFW_IMGUI ON)
#set(CINOLIB_USES_TRIANGLE ON)
#set(CINOLIB_USES_TETGEN ON)
set(CINOLIB_USES_SPECTRA ON)
find_package(cinolib REQUIRED)
add_library(ANN STATIC IMPORTED)
set_property(TARGET ANN PROPERTY IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/ann_1.1.2/lib/libANN.a)
target_link_libraries(${PROJECT_NAME} cinolib ANN)
target_include_directories(${PROJECT_NAME} PUBLIC ${cinolib_DIR}/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/libs)

add_executable(smooth_shape mean_curv.cpp diff_geo.h diff_geo.cpp utilities.h utilities.cpp drawing_circle.h drawing_circle.cpp)
set_target_properties(smooth_shape    PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
include_directories(${CMAKE_SOURCE_DIR}/libs/Eigen)
include_directories(${CMAKE_SOURCE_DIR}/libs/ann_1.1.2/include)
target_link_libraries(smooth_shape cinolib ANN)
target_include_directories(smooth_shape PUBLIC ${cinolib_DIR}/include)
target_include_directories(smooth_shape PUBLIC ${CMAKE_SOURCE_DIR}/libs)


