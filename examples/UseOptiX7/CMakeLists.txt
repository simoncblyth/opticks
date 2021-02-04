cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOptiX7)
project(${name})

find_package(OpticksCUDA  MODULE QUIET)   
find_package(OptiX REQUIRED)   

#[=[
Notice no OptiX libs anymore in 7, 
just headers giving access to functionality from the driver ?
#]=]


include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OptiX_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR} 
)

add_executable(${name} ${name}.cc)
target_link_libraries(${name} 
     PUBLIC 
         ${CUDA_LIBRARIES}
)
install(TARGETS ${name}  DESTINATION lib)


