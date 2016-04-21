#include(OPTICKSCfg)

set(OGLRap_PREFIX "${OPTICKS_PREFIX}/graphics/oglrap")

find_library( OGLRap_LIBRARIES 
              NAMES OGLRap
              PATHS ${OGLRap_PREFIX}/lib )

set(OGLRap_INCLUDE_DIRS "${OGLRap_PREFIX}/include")
set(OGLRap_DEFINITIONS "")

