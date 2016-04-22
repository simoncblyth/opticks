find_library( OptiXRap_LIBRARIES 
              NAMES OptiXRap
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT OptiXRap_LIBRARIES)
       set(OptiXRap_LIBRARIES OptiXRap)
    endif()
endif(SUPERBUILD)


set(OptiXRap_INCLUDE_DIRS "${OptiXRap_SOURCE_DIR}")
set(OptiXRap_DEFINITIONS "")

