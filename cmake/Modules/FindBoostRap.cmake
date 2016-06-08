find_library( BoostRap_LIBRARIES 
              NAMES BoostRap
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT BoostRap_LIBRARIES)
       set(BoostRap_LIBRARIES BoostRap)
    endif()
endif(SUPERBUILD)

# find_package normally yields NOTFOUND
# when no lib is found at configure time : ie when cmake is run
# but here if SUPERBUILD is defined BRegex_LIBRARIES
# is set to BRegex the target name, 
# which will allow the build to succeed if the target
# is included amongst the add_subdirectory of the super build

set(BoostRap_INCLUDE_DIRS "${BoostRap_SOURCE_DIR}")

set(BoostRap_DEFINITIONS "")

