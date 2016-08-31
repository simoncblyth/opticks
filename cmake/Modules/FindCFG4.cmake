find_library( CFG4_LIBRARIES 
              NAMES cfg4
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT CFG4_LIBRARIES)
       set(CFG4_LIBRARIES cfg4)
    endif()
endif(SUPERBUILD)


# projname_SOURCE_DIR : projname must exactly  match the project name
set(CFG4_INCLUDE_DIRS "${cfg4_SOURCE_DIR}")
set(CFG4_DEFINITIONS "")

