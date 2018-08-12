CUDA config
=============

Formerly has to do something like the below in the CMakeLists.txt of 
packages that use CUDA::

    cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
    set(name UseOpticksCUDA)
    project(${name} VERSION 0.1.0)
    include(OpticksBuildOptions)

    find_package(OKConf REQUIRED CONFIG)   
    message(STATUS "${name}.COMPUTE_CAPABILITY ${COMPUTE_CAPABILITY} ")

    set(FLAGS_VERBOSE ON) 
    include(OpticksCompilationFlags)  

    #[=[
    OpticksCompilationFlags comes with OpticksBuildOptions, 
    however when CUDA is in use need to repeat the include
    after OKConf, which defines COMPUTE_CAPABILITY, 
    for correct setup of nvcc flags 
    #]=]


    set(OpticksCUDA_VERBOSE ON) 
    find_package(OpticksCUDA REQUIRED MODULE) 

    cuda_add_executable(${name} ${name}.cu)

    target_link_libraries(${name} Opticks::CUDA Opticks::CUDASamples)

    install(TARGETS ${name} DESTINATION lib)


To try to avoid this annoyance have split OpticksCompilationFlags.cmake into::

    OpticksCXXFlags.cmake
    OpticksCUDAFlags.cmake
 
and changed the generated TOPMATTER of OKConf to include OpticksCUDAFlags.
In this way I hope nothing special should be needed to use CUDA now... 
so long as the CMakeLists.txt of the package does a find_package on OKConf
or some other package the dependency tree of which includes OKConf.

::

    086 
     87 
     88 bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL TOPMATTER "
     89 ## OKConf generated TOPMATTER
     90 
     91 set(OptiX_INSTALL_DIR ${OptiX_INSTALL_DIR})
     92 set(COMPUTE_CAPABILITY ${COMPUTE_CAPABILITY})
     93 
     94 if(OKConf_VERBOSE)
     95   message(STATUS \"\${CMAKE_CURRENT_LIST_FILE} : OKConf_VERBOSE     : \${OKConf_VERBOSE} \")
     96   message(STATUS \"\${CMAKE_CURRENT_LIST_FILE} : OptiX_INSTALL_DIR  : \${OptiX_INSTALL_DIR} \")
     97   message(STATUS \"\${CMAKE_CURRENT_LIST_FILE} : COMPUTE_CAPABILITY : \${COMPUTE_CAPABILITY} \")
     98 endif()
     99 
    100 include(OpticksCUDAFlags)
    101 
    102 " )



OpticksCompilationFlags no longer exists, so need to fix these::

    epsilon:Modules blyth$ opticks-find OpticksCompilationFlags
    ./okop/CMakeLists.txt:include(OpticksCompilationFlags)      
    ./cudarap/CMakeLists.txt:include(OpticksCompilationFlags)     ## repeating this here as nvcc flags require COMPUTE_CAPABILIY from OKConf     

    ./optickscore/CMakeLists.txt:Were still getting some of the above warning following addition of -fvisibility-inlines-hidden to OpticksCompilationFlags.cmake.
    ./thrustrap/CMakeLists.txt:include(OpticksCompilationFlags)
    ./examples/UseOpticksCUDA/CMakeLists.txt:include(OpticksCompilationFlags)  
    ./examples/UseOpticksCUDA/CMakeLists.txt:OpticksCompilationFlags comes with OpticksBuildOptions, 
    ./examples/UseOptiXProgram/CMakeLists.txt:include(OpticksCompilationFlags)     ## repeating this here as nvcc flags require COMPUTE_CAPABILIY from OKConf     
    ./examples/ThrustOpenGLInterop/CMakeLists.txt:include(OpticksCompilationFlags)     ## repeating this here as nvcc flags require COMPUTE_CAPABILIY from OKConf  
    ./boostrap/CMakeLists.txt:Avoided this warning by adding to OpticksCompilationFlags.cmake::
    ./optixrap/CMakeLists.txt:include(OpticksCompilationFlags)   


