
find_library( OptiXThrust_LIBRARIES 
              NAMES OptiXThrust
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT OptiXThrust_LIBRARIES)
       set(OptiXThrust_LIBRARIES OptiXThrust)
    endif()
endif(SUPERBUILD)


set(OptiXThrust_INCLUDE_DIRS "${OptiXThrust_SOURCE_DIR}")
set(OptiXThrust_DEFINITIONS "")

