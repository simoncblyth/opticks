
find_library( OptiXThrust_LIBRARIES 
              NAMES OptiXThrust
              PATHS ${OPTICKS_PREFIX}/lib )

#set(OptiXThrust_INCLUDE_DIRS "${OPTICKS_PREFIX}/include/OptiXThrust")
set(OptiXThrust_INCLUDE_DIRS "${OPTICKS_HOME}/optix/optixthrust")
set(OptiXThrust_DEFINITIONS "")

