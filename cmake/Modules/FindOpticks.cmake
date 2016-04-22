find_library( Opticks_LIBRARIES 
              NAMES Opticks
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT Opticks_LIBRARIES)
       set(Opticks_LIBRARIES Opticks)
    endif()
endif(SUPERBUILD)

set(Opticks_INCLUDE_DIRS "${Opticks_SOURCE_DIR}")

set(Opticks_DEFINITIONS "")

