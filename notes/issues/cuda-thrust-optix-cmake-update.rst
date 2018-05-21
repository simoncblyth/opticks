CUDA Thrust OptiX CMake Update ?
===================================

Modern CMake 3.9+ has new possibilities...
----------------------------------------

* https://devblogs.nvidia.com/building-cuda-applications-cmake/

* http://on-demand.gputechconf.com/gtc/2017/presentation/S7438-robert-maynard-build-systems-combining-cuda-and-machine-learning.pdf

  ~/opticks_refs/S7438-robert-maynard-build-systems-combining-cuda-and-machine-learning.pdf

Build Systems: Combining CUDA and Modern CMake
GTC, San Jose, CA May, 2017
Robert Maynard


CMake 3.9: PTX
::

   add_library(CudaPTXObjects OBJECT kernelA.cu kernelB.cu)
   set_property(TARGET CudaPTXObjects PROPERTY CUDA_PTX_COMPILATION ON)
   target_compile_definitions( CudaPTXObjects PUBLIC "PTX_FILES")


* https://gitlab.kitware.com/robertmaynard/cmake_cuda_tests


* https://stackoverflow.com/questions/46903060/cmake-3-8-setting-different-compiler-flags-for-projects-that-include-both-cpp


caffe2 has a go at making targets

* https://github.com/caffe2/caffe2/blob/master/cmake/public/cuda.cmake


Old way of using CUDA relies on " CUDA_ADD_LIBRARY/CUDA_ADD_EXECUTABLE
------------------------------------------------------------------------

::

    epsilon:UseCUDA blyth$ port contents cmake | grep FindCUDA.cmake
      /opt/local/share/cmake-3.11/Modules/FindCUDA.cmake

     160 #   CUDA_ADD_EXECUTABLE( cuda_target file0 file1 ...
     161 #                        [WIN32] [MACOSX_BUNDLE] [EXCLUDE_FROM_ALL] [OPTIONS ...] )
     162 #   -- Creates an executable "cuda_target" which is made up of the files
     163 #      specified.  All of the non CUDA C files are compiled using the standard
     164 #      build rules specified by CMAKE and the cuda files are compiled to object
     165 #      files using nvcc and the host compiler.  In addition CUDA_INCLUDE_DIRS is
     166 #      added automatically to include_directories().  Some standard CMake target
     167 #      calls can be used on the target after calling this macro
     168 #      (e.g. set_target_properties and target_link_libraries), but setting
     169 #      properties that adjust compilation flags will not affect code compiled by
     170 #      nvcc.  Such flags should be modified before calling CUDA_ADD_EXECUTABLE,
     171 #      CUDA_ADD_LIBRARY or CUDA_WRAP_SRCS.
     172 #
     173 #   CUDA_ADD_LIBRARY( cuda_target file0 file1 ...
     174 #                     [STATIC | SHARED | MODULE] [EXCLUDE_FROM_ALL] [OPTIONS ...] )
     175 #   -- Same as CUDA_ADD_EXECUTABLE except that a library is created.
     176 #




    epsilon:UseCUDA blyth$ cmake --help-module FindCUDA
    FindCUDA
    --------

    .. note::

      The FindCUDA module has been superseded by first-class support
      for the CUDA language in CMake.  It is no longer necessary to
      use this module or call ``find_package(CUDA)``.  This module
      now exists only for compatibility with projects that have not
      been ported.




::

     62 function(optixthrust_add_executable target_name)
     63 
     64     # split arguments into four lists 
     65     #  hmm have two different flavors of .cu
     66     #  optix programs to be made into .ptx  
     67     #  and thrust or CUDA non optix sources need to be compiled into .o for linkage
     68 
     69     OPTIXTHRUST_GET_SOURCES_AND_OPTIONS(optix_source_files non_optix_source_files cmake_options options ${ARGN})
     70 
     71     #message( "OPTIXTHRUST:optix_source_files= " "${optix_source_files}" )  
     72     #message( "OPTIXTHRUST:non_optix_source_files= "  "${non_optix_source_files}" )  
     73 
     74     # Create the rules to build the OBJ from the CUDA files.
     75     #message( "OPTIXTHRUST:OBJ options = " "${options}" )  
     76     CUDA_WRAP_SRCS( ${target_name} OBJ non_optix_generated_files ${non_optix_source_files} ${cmake_options} OPTIONS ${options} )
     77 
     78     # Create the rules to build the PTX from the CUDA files.
     79     #message( "OPTIXTHRUST:PTX options = " "${options}" )  
     80     CUDA_WRAP_SRCS( ${target_name} PTX optix_generated_files ${optix_source_files} ${cmake_options} OPTIONS ${options} )
     81 
     82     add_executable(${target_name}
     83         ${optix_source_files}
     84         ${non_optix_source_files}
     85         ${optix_generated_files}
     86         ${non_optix_generated_files}
     87         ${cmake_options}
     88     )
     89 
     90     target_link_libraries( ${target_name}
     91         ${LIBRARIES}
     92       )
     93 
     94 endfunction()


::

    099 function(optixthrust_add_library target_name)
    100 
    101     # split arguments into four lists 
    102     #  hmm have two different flavors of .cu
    103     #  optix programs to be made into .ptx  
    104     #  and thrust or CUDA non optix sources need to be compiled into .o for linkage
    105 
    106     OPTIXTHRUST_GET_SOURCES_AND_OPTIONS(optix_source_files non_optix_source_files cmake_options options ${ARGN})
    107 
    108     #message( "OPTIXTHRUST:optix_source_files= " "${optix_source_files}" )  
    109     #message( "OPTIXTHRUST:non_optix_source_files= "  "${non_optix_source_files}" )  
    110 
    111     # Create the rules to build the OBJ from the CUDA files.
    112     #message( "OPTIXTHRUST:OBJ options = " "${options}" )  
    113     CUDA_WRAP_SRCS( ${target_name} OBJ non_optix_generated_files ${non_optix_source_files} ${cmake_options} OPTIONS ${options} )
    114 
    115     # Create the rules to build the PTX from the CUDA files.
    116     #message( "OPTIXTHRUST:PTX options = " "${options}" )  
    117     CUDA_WRAP_SRCS( ${target_name} PTX optix_generated_files ${optix_source_files} ${cmake_options} OPTIONS ${options} )
    118 
    119     add_library(${target_name}
    120         ${optix_source_files}
    121         ${non_optix_source_files}
    122         ${optix_generated_files}
    123         ${non_optix_generated_files}
    124         ${cmake_options}
    125     )
    126 
    127     target_link_libraries( ${target_name}
    128         ${LIBRARIES}
    129       )
    130 
    131 endfunction()
    132 
    133 
    134 # if cmake variable CUDA_GENERATED_OUTPUT_DIR is
    135 # defined then both OBJ and PTX output is lumped 
    136 # together in that directory, prefer to not defining
    137 # it in order for different directories to be used



