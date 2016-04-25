find_library( BRegex_LIBRARIES 
              NAMES BRegex
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT BRegex_LIBRARIES)
       set(BRegex_LIBRARIES BRegex)
    endif()
endif(SUPERBUILD)

# find_package normally yields NOTFOUND
# when no lib is found at configure time : ie when cmake is run
# but here if SUPERBUILD is defined BRegex_LIBRARIES
# is set to BRegex the target name, 
# which will allow the build to succeed if the target
# is included amongst the add_subdirectory of the super build

set(BRegex_INCLUDE_DIRS "${BRegex_SOURCE_DIR}")

set(BRegex_DEFINITIONS "")

