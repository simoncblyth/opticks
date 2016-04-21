#include(OPTICKSCfg)
set(NPY_PREFIX "${OPTICKS_PREFIX}/numerics/npy")

find_library( NPY_LIBRARIES 
              NAMES NPY
              PATHS ${NPY_PREFIX}/lib )

set(NPY_INCLUDE_DIRS "${NPY_PREFIX}/include")
set(NPY_DEFINITIONS "")

