find_library( CUDAWrap_LIBRARIES 
              NAMES CUDAWrap
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT CUDAWrap_LIBRARIES)
       set(CUDAWrap_LIBRARIES CUDAWrap)
    endif()
endif(SUPERBUILD)


set(CUDAWrap_INCLUDE_DIRS "${CUDAWrap_SOURCE_DIR}")
set(CUDAWrap_DEFINITIONS "")

