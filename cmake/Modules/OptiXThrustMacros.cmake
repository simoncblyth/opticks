#  Depends on:
#
#  * FindOptiX and FindCUDA (developed against those from OptiX 3.8)
#    include this after finding those
#
#  * a LIBRARIES variable listing libraries to be linked
#
##############################################################################
#
# modified /Developer/OptiX/SDK/CMake/FindCUDA.cmake:CUDA_GET_SOURCES_AND_OPTIONS 
# to partion .cu into flavors:
#
# * OptiX RTProgram to be compiled to .ptx for runtime loading/compilation by OptiX 
# * vanilla CUDA/Thrust for compilation to .o for normal linkage 
#
# Separate the OPTIONS out from the sources
#
macro(OPTIXTHRUST_GET_SOURCES_AND_OPTIONS _optix_sources _non_optix_sources _cmake_options _options)
  set( ${_optix_sources} )
  set( ${_non_optix_sources} )
  set( ${_cmake_options} )
  set( ${_options} )
  set( _found_options FALSE )
  foreach(arg ${ARGN})
    message(${arg})
    if(arg STREQUAL "OPTIONS")
      set( _found_options TRUE )
    elseif(
        arg STREQUAL "WIN32" OR
        arg STREQUAL "MACOSX_BUNDLE" OR
        arg STREQUAL "EXCLUDE_FROM_ALL" OR
        arg STREQUAL "STATIC" OR
        arg STREQUAL "SHARED" OR
        arg STREQUAL "MODULE"
        )
      list(APPEND ${_cmake_options} ${arg})
    else()
      if ( _found_options )
        list(APPEND ${_options} ${arg})
      else()

        # Assume this is a file, flavor of .cu is detected by its placement:
        #
        #   * in the immediate directory for non optix CUDA/Thrust .cu   
        #   * in subdirectories for optix program .cu
        # 

        set(_match_non_optix_cu) 
        string(REGEX MATCH "^[^/]+[.]cu$" _match_non_optix_cu ${arg})
        if(_match_non_optix_cu)
           list(APPEND ${_non_optix_sources} ${arg})
        else()
           list(APPEND ${_optix_sources} ${arg})
        endif()
      endif()
    endif()
  endforeach()
endmacro()



function(optixthrust_add_executable target_name)

    # split arguments into four lists 
    #  hmm have two different flavors of .cu
    #  optix programs to be made into .ptx  
    #  and thrust or CUDA non optix sources need to be compiled into .o for linkage

    OPTIXTHRUST_GET_SOURCES_AND_OPTIONS(optix_source_files non_optix_source_files cmake_options options ${ARGN})

    message( "optix_source_files= " "${optix_source_files}" )  
    message( "non_optix_source_files= "  "${non_optix_source_files}" )  

    # Create the rules to build the OBJ from the CUDA files.
    message( "OBJ options = " "${options}" )  
    CUDA_WRAP_SRCS( ${target_name} OBJ non_optix_generated_files ${non_optix_source_files} ${cmake_options} OPTIONS ${options} )

    # Create the rules to build the PTX from the CUDA files.
    message( "PTX options = " "${options}" )  
    CUDA_WRAP_SRCS( ${target_name} PTX optix_generated_files ${optix_source_files} ${cmake_options} OPTIONS ${options} )

    add_executable(${target_name}
        ${optix_source_files}
        ${non_optix_source_files}
        ${optix_generated_files}
        ${non_optix_generated_files}
        ${cmake_options}
    )

    target_link_libraries( ${target_name} 
        ${LIBRARIES} 
      )

endfunction()




function(optixthrust_add_library target_name)

    # split arguments into four lists 
    #  hmm have two different flavors of .cu
    #  optix programs to be made into .ptx  
    #  and thrust or CUDA non optix sources need to be compiled into .o for linkage

    OPTIXTHRUST_GET_SOURCES_AND_OPTIONS(optix_source_files non_optix_source_files cmake_options options ${ARGN})

    message( "optix_source_files= " "${optix_source_files}" )  
    message( "non_optix_source_files= "  "${non_optix_source_files}" )  

    # Create the rules to build the OBJ from the CUDA files.
    message( "OBJ options = " "${options}" )  
    CUDA_WRAP_SRCS( ${target_name} OBJ non_optix_generated_files ${non_optix_source_files} ${cmake_options} OPTIONS ${options} )

    # Create the rules to build the PTX from the CUDA files.
    message( "PTX options = " "${options}" )  
    CUDA_WRAP_SRCS( ${target_name} PTX optix_generated_files ${optix_source_files} ${cmake_options} OPTIONS ${options} )

    add_library(${target_name}
        ${optix_source_files}
        ${non_optix_source_files}
        ${optix_generated_files}
        ${non_optix_generated_files}
        ${cmake_options}
    )

    target_link_libraries( ${target_name} 
        ${LIBRARIES} 
      )

endfunction()


# if cmake variable CUDA_GENERATED_OUTPUT_DIR is
# defined then both OBJ and PTX output is lumped 
# together in that directory, prefer to not defining
# it in order for different directories to be used
#
# needs to match that assumed/configured within RayTraceConfig.cc
#set(CUDA_GENERATED_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib/ptx") 
#message("CUDA_GENERATED_OUTPUT_DIR:" ${CUDA_GENERATED_OUTPUT_DIR}) 
#
