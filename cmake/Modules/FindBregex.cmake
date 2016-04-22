find_library( Bregex_LIBRARIES 
              NAMES Bregex
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT Bregex_LIBRARIES)
       set(Bregex_LIBRARIES Bregex)
    endif()
endif(SUPERBUILD)

# find_package normally yields NOTFOUND
# when no lib is found at configure time : ie when cmake is run
# but here if SUPERBUILD is defined Bregex_LIBRARIES
# is set to Bregex the target name, 
# which will allow the build to succeed if the target
# is included amongst the add_subdirectory of the super build

set(Bregex_INCLUDE_DIRS "${Bregex_SOURCE_DIR}")

set(Bregex_DEFINITIONS "")

