project(gpu_sky_simulator)
cmake_minimum_required(VERSION 3.5)
# enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES "61")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall")

#find_package(OpenCV REQUIRED)
find_package(Pangolin 0.8 REQUIRED)
#find_package(Pangolin REQUIRED)
#find_package(MetavisionSDK COMPONENTS cv core driver REQUIRED)


include_directories( 
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/extern/perturb/include
    /usr/include/eigen3
    ${Pangolin_INCLUDE_DIRS}
)

link_directories(
    ${PROJECT_SOURCE_DIR}/extern/perturb/lib
)

# add_executable(sim_gpu src/sim_gpu.cu)
# target_link_libraries(sim_gpu pango_display pango_python)

# add_executable(sim_cpu src/sim_cpu.cpp)
# target_link_libraries(sim_cpu pango_display pango_python)

add_executable(sim_sgp4 src/sim_sgp4.cpp)
target_link_libraries(sim_sgp4 pango_display pango_python perturb)



# add_executable(sim_cli src/sim_cli.cpp)
# target_link_libraries(sim_cli pango_display)