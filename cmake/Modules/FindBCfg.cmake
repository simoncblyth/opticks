find_library( BCfg_LIBRARIES 
              NAMES BCfg
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT BCfg_LIBRARIES)
       set(BCfg_LIBRARIES BCfg)
    endif()
endif(SUPERBUILD)



set(BCfg_INCLUDE_DIRS "${BCfg_SOURCE_DIR}")

set(BCfg_DEFINITIONS "")

