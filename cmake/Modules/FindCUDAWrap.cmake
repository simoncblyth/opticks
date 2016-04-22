find_library( CUDAWrap_LIBRARIES 
              NAMES CUDAWrap
              PATHS ${OPTICKS_PREFIX}/lib )

#set(CUDAWrap_INCLUDE_DIRS "${OPTICKS_PREFIX}/include/CUDAWrap")
set(CUDAWrap_INCLUDE_DIRS "${OPTICKS_HOME}/cuda/cudawrap")
set(CUDAWrap_DEFINITIONS "")

