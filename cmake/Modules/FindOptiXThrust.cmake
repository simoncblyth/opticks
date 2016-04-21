#include(OPTICKSCfg)

set(OptiXThrust_PREFIX "${OPTICKS_PREFIX}/graphics/optixthrust")

find_library( OptiXThrust_LIBRARIES 
              NAMES OptiXThrustMinimalLib
              PATHS ${OptiXThrust_PREFIX}/lib )

set(OptiXThrust_INCLUDE_DIRS "${OptiXThrust_PREFIX}/include")
set(OptiXThrust_DEFINITIONS "")

