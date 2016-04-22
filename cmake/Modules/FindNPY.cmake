find_library( NPY_LIBRARIES 
              NAMES NPY
              PATHS ${OPTICKS_PREFIX}/lib )

#set(NPY_INCLUDE_DIRS "${OPTICKS_PREFIX}/include/NPY")
set(NPY_INCLUDE_DIRS "${OPTICKS_HOME}/numerics/npy")

set(NPY_DEFINITIONS "")

