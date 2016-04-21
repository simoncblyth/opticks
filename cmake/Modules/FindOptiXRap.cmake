#include(OPTICKSCfg)

set(OptiXRap_PREFIX "${OPTICKS_PREFIX}/graphics/OptiXRap")

find_library( OptiXRap_LIBRARIES 
              NAMES OptiXRap
              PATHS ${OptiXRap_PREFIX}/lib )

set(OptiXRap_INCLUDE_DIRS "${OptiXRap_PREFIX}/include")
set(OptiXRap_DEFINITIONS "")

