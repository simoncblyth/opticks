find_library( OpticksCore_LIBRARIES 
              NAMES OpticksCore
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT OpticksCore_LIBRARIES)
       set(OpticksCore_LIBRARIES OpticksCore)
    endif()
endif(SUPERBUILD)

set(OpticksCore_INCLUDE_DIRS "${OpticksCore_SOURCE_DIR}")

set(OpticksCore_DEFINITIONS "")

