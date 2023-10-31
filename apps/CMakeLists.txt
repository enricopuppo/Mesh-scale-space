
add_executable(reg_shapes main.cpp app.h app.cpp)
add_definitions(-DSHADERS_PATH="${PROJECT_SOURCE_DIR}/shaders/")
add_definitions(-DPROJECT_PATH="${PROJECT_SOURCE_DIR}/")
set_target_properties(reg_shapes     PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
target_include_directories(reg_shapes    PUBLIC ${CMAKE_SOURCE_DIR}/libs)
target_link_libraries(reg_shapes yocto diff_geo yocto_gui realtime utils)
