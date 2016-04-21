#include(OPTICKSCfg)

set(CUDAWrap_PREFIX "${OPTICKS_PREFIX}/cuda/CUDAWrap")

find_library( CUDAWrap_LIBRARIES 
              NAMES CUDAWrap
              PATHS ${CUDAWrap_PREFIX}/lib )

set(CUDAWrap_INCLUDE_DIRS "${CUDAWrap_PREFIX}/include")
set(CUDAWrap_DEFINITIONS "")

