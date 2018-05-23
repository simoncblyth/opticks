find_library( OK_LIBRARIES 
              NAMES OK
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT OK_LIBRARIES)
       set(OK_LIBRARIES OK)
    endif()
endif(SUPERBUILD)


set(OK_INCLUDE_DIRS "${OK_SOURCE_DIR}")
set(OK_DEFINITIONS "")

