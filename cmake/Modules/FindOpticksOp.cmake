find_library( OpticksOp_LIBRARIES 
              NAMES OpticksOp
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT OpticksOp_LIBRARIES)
       set(OpticksOp_LIBRARIES OpticksOp)
    endif()
endif(SUPERBUILD)


set(OpticksOp_INCLUDE_DIRS "${OpticksOp_SOURCE_DIR}")
set(OpticksOp_DEFINITIONS "")

