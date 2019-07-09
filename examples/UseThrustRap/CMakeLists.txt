cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseThrustRap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)


if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
    cmake_policy(SET CMP0077 OLD)
endif()



find_package(ThrustRap REQUIRED CONFIG)

echo_target_std(Opticks::ThrustRap)

cuda_add_executable( ${name} ${name}.cu )

target_link_libraries(${name} Opticks::ThrustRap )
# TODO: check on this plain signature vs the PUBLIC one 
# have to use plain as cuda_add_executable does so internally
# although maybe can control via CUDA*KEYWORD var 

install(TARGETS ${name} DESTINATION lib)

