find_library( OptiXRap_LIBRARIES 
              NAMES OptiXRap
              PATHS ${OPTICKS_PREFIX}/lib )

#set(OptiXRap_INCLUDE_DIRS "${OPTICKS_PREFIX}/include/OptiXRap")
set(OptiXRap_INCLUDE_DIRS "${OPTICKS_HOME}/graphics/optixrap")
set(OptiXRap_DEFINITIONS "")

