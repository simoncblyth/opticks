#include(OPTICKSCfg)
set(AssimpWrap_PREFIX "${OPTICKS_PREFIX}/graphics/assimpwrap")

find_library( AssimpWrap_LIBRARIES 
              NAMES AssimpWrap
              PATHS ${AssimpWrap_PREFIX}/lib )

set(AssimpWrap_INCLUDE_DIRS "${AssimpWrap_PREFIX}/include")
set(AssimpWrap_DEFINITIONS "")

