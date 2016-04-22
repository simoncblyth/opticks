find_library( Cfg_LIBRARIES 
              NAMES Cfg
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT Cfg_LIBRARIES)
       set(Cfg_LIBRARIES Cfg)
    endif()
endif(SUPERBUILD)



set(Cfg_INCLUDE_DIRS "${Cfg_SOURCE_DIR}")

set(Cfg_DEFINITIONS "")

