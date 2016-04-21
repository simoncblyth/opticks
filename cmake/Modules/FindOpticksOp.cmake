#include(OPTICKSCfg)

set(OpticksOp_PREFIX "${OPTICKS_PREFIX}/opticksop")

find_library( OpticksOp_LIBRARIES 
              NAMES OpticksOp
              PATHS ${OpticksOp_PREFIX}/lib )

set(OpticksOp_INCLUDE_DIRS "${OpticksOp_PREFIX}/include")
set(OpticksOp_DEFINITIONS "")

